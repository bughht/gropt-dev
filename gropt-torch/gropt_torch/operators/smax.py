import torch
from .base import Operator


class Op_Smax(Operator):
    def __init__(self, smax: float, rot_variant: bool, weight_mod: float = 1.0):
        super().__init__(name="Slew", weight_mod=weight_mod)
        self.smax = float(smax)
        self.rot_variant = bool(rot_variant)

    def init(self, pdata) -> None:
        self.target = 0.0
        self.tol0 = self.smax
        self.tol = (1.0 - self.cushion) * self.tol0

        self.spec_norm2 = 4.0 / pdata.dt / pdata.dt
        self.spec_norm = float(self.spec_norm2) ** 0.5

        self.Ax_size = pdata.Naxis * (pdata.N - 1)

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * X).
        Calculates the slew rate by taking the discrete derivative (differences)
        between adjacent gradient points. This is a purely linear operation.
        """
        n = self.N
        x_mat = x.view(self.Naxis, n)
        out = torch.diff(x_mat, dim=1) / self.dt
        return out.reshape(-1)

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_i^T * X).
        The transpose of the discrete derivative matrix. It acts like a negative
        divergence operator, moving the boundary forces back onto the gradient nodes.
        """
        n = self.N
        x_mat = x.view(self.Naxis, n - 1)
        out = torch.zeros((self.Naxis, n), dtype=x.dtype, device=x.device)
        out[:, 0] = -x_mat[:, 0] / self.dt
        out[:, 1:-1] = (x_mat[:, :-1] - x_mat[:, 1:]) / self.dt
        out[:, -1] = x_mat[:, -1] / self.dt
        return out.reshape(-1)

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Geometry (Proximal Hard Constraint).
        Clips any slew rate that exceeds the Smax boundary.
        - Rotational variant: Hard clamp on each axis independently.
        - Rotational invariant: Radial compression of the 3D slew vector to fit within the Smax sphere.
        """
        x = x.clone()

        if self.do_equil:
            x = x / self.eq_rows
        x = x * self.spec_norm

        if self.rot_variant:
            lower = self.target - self.tol
            upper = self.target + self.tol
            x = torch.clamp(x, min=lower, max=upper)
        else:
            n = self.N
            upper = self.target + self.tol
            x_mat = x.view(self.Naxis, n - 1)
            norms = torch.linalg.norm(x_mat, dim=0)
            mask = norms > upper
            if torch.any(mask):
                scale = upper / (norms + 1.0e-32)
                x_mat[:, mask] = x_mat[:, mask] * scale[mask]
            x = x_mat.reshape(-1)

        if self.do_equil:
            x = x * self.eq_rows
        x = x / self.spec_norm

        return x

    def check(self, x: torch.Tensor) -> None:
        is_feas = 1

        if self.do_equil:
            x = x / self.eq_rows
        x = x * self.spec_norm

        if self.rot_variant:
            lower = self.target - self.tol0
            upper = self.target + self.tol0
            mask = torch.isnan(self.pdata.set_vals[: x.numel()])
            mask_prev = torch.roll(mask, 1)
            mask_next = torch.roll(mask, -1)
            mask_prev[0] = mask[0]
            mask_next[-1] = mask[-1]
            should_check = mask | mask_prev | mask_next
            infeas = ((x < lower) | (x > upper)) & should_check
            if torch.any(infeas):
                is_feas = 0
        else:
            n = self.N
            upper = self.target + self.tol0
            x_mat = x.view(self.Naxis, n - 1)
            norms = torch.linalg.norm(x_mat, dim=0)
            base_mask = torch.isnan(self.pdata.set_vals[:n])
            mask = base_mask[: n - 1]
            mask_prev = torch.roll(base_mask, 1)[: n - 1]
            mask_next = torch.roll(base_mask, -1)[: n - 1]
            mask_prev[0] = base_mask[0]
            mask_next[-1] = base_mask[-1]
            should_check = mask | mask_prev | mask_next
            infeas = (norms > upper) & should_check
            if torch.any(infeas):
                is_feas = 0

        if self.do_equil:
            x = x * self.eq_rows
        x = x / self.spec_norm

        self.hist_feas.append(is_feas)
