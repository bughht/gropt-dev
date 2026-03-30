import torch
from .base import Operator


class Op_TV(Operator):
    def __init__(self, tv_lam: float = 0.0, weight_mod: float = 1.0):
        super().__init__(name="TotalVariation", weight_mod=weight_mod)
        self.tv_lam = float(tv_lam)

    def init(self, pdata) -> None:
        self.target = 0.0
        self.tol0 = self.tv_lam
        self.tol = (1.0 - self.cushion) * self.tol0

        self.spec_norm2 = 4.0 / pdata.dt / pdata.dt
        self.spec_norm = float(self.spec_norm2) ** 0.5

        self.Ax_size = pdata.Naxis * (pdata.N - 1)

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator.
        Calculates the discrete derivative (slew) or differences between adjacent points.
        This provides the linear mapping needed for the Total Variation constraint.
        """
        x_mat = x.view(self.Naxis, self.N)
        out = torch.diff(x_mat, dim=1) / self.dt
        return out.reshape(-1)

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator.
        Performs the adjoint of the discrete differences (similar to negative divergence).
        Coupled with the forward operator, it builds the curvature for the CG solver.
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
        The Non-Linear Proximal Operator (Soft Thresholding).
        Applies L1 regularization by shrinking the magnitudes of the differences
        towards zero by `tv_lam`, which encourages a piecewise constant (staircase) waveform.
        """
        x = x.clone()
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)
        x = torch.where(abs_x > self.tv_lam, sign_x * (abs_x - self.tv_lam), torch.zeros_like(x))
        return x

    def check(self, x: torch.Tensor) -> None:
        self.hist_feas.append(1)
