import torch
from .base import Operator


class Op_Gmax(Operator):
    def __init__(self, gmax: float, rot_variant: bool, weight_mod: float = 1.0):
        super().__init__(name="Gradient", weight_mod=weight_mod)
        self.gmax = float(gmax)
        self.rot_variant = bool(rot_variant)

    def init(self, pdata) -> None:
        self.target = 0.0
        self.tol0 = self.gmax
        self.tol = (1.0 - self.cushion) * self.tol0

        self.spec_norm2 = 1.0
        self.spec_norm = 1.0

        self.Ax_size = pdata.Naxis * pdata.N

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * X).
        For maximum gradient amplitude (Gmax), the operator is just the Identity matrix.
        So we just return the waveform itself.
        """
        return x.clone()

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_i^T * X).
        The transpose of the Identity matrix is just the Identity matrix.
        """
        return x.clone()

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Geometry (Proximal Hard Constraint).
        This clips any gradient amplitude that exceeds the Gmax limit.
        - Rotational variant: Clips each axis (x, y, z) independently to a box.
        - Rotational invariant: Calculates the L2 norm across axes and radially compresses the vector into a sphere.
        """
        x = x.clone()

        if self.do_equil:
            x = x / self.eq_rows

        if self.rot_variant:
            lower = self.target - self.tol
            upper = self.target + self.tol
            x = torch.clamp(x, min=lower, max=upper)

            if self.pdata is not None:
                mask_fixed = ~torch.isnan(self.pdata.set_vals)
                x[mask_fixed] = self.pdata.set_vals[mask_fixed]
        else:
            n = self.N
            n_axis = self.Naxis
            x_mat = x.view(n_axis, n)
            upper = self.target + self.tol
            norms = torch.linalg.norm(x_mat, dim=0)
            mask = norms > upper
            if torch.any(mask):
                scale = upper / (norms + 1.0e-32)
                x_mat[:, mask] = x_mat[:, mask] * scale[mask]

            x = x_mat.reshape(-1)
            if self.pdata is not None:
                mask_fixed = ~torch.isnan(self.pdata.set_vals)
                x[mask_fixed] = self.pdata.set_vals[mask_fixed]

        if self.do_equil:
            x = x * self.eq_rows

        return x

    def check(self, x: torch.Tensor) -> None:
        is_feas = 1

        if self.do_equil:
            x = x / self.eq_rows

        if self.rot_variant:
            lower = self.target - self.tol0
            upper = self.target + self.tol0
            mask_free = torch.isnan(self.pdata.set_vals)
            infeas = (x < lower) | ((x > upper) & mask_free)
            if torch.any(infeas):
                is_feas = 0
        else:
            n = self.N
            n_axis = self.Naxis
            upper = self.target + self.tol0
            x_mat = x.view(n_axis, n)
            norms = torch.linalg.norm(x_mat, dim=0)
            mask_free = torch.isnan(self.pdata.set_vals[:n])
            infeas = (norms > upper) & mask_free
            if torch.any(infeas):
                is_feas = 0

        if self.do_equil:
            x = x * self.eq_rows

        self.hist_feas.append(is_feas)
