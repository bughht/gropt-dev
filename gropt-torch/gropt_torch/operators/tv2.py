import torch
from .base import Operator


class Op_TV2(Operator):
    def __init__(self, tv2_lam: float = 0.0, weight_mod: float = 1.0):
        super().__init__(name="TotalVariation2", weight_mod=weight_mod)
        self.tv2_lam = float(tv2_lam)

    def init(self, pdata) -> None:
        self.target = 0.0
        self.tol0 = self.tv2_lam
        self.tol = (1.0 - self.cushion) * self.tol0

        self.spec_norm2 = 16.0 / (pdata.dt**4)
        self.spec_norm = float(self.spec_norm2) ** 0.5

        self.Ax_size = pdata.Naxis * (pdata.N - 2)

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_2 * X).
        Second derivative to penalize slew rate changes.
        """
        x_mat = x.view(self.Naxis, self.N)
        out = torch.diff(x_mat, n=2, dim=1) / (self.dt ** 2)
        return out.reshape(-1)

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_2^T * X).
        """
        n = self.N
        n_2 = n - 2
        x_mat = x.view(self.Naxis, n_2)
        out = torch.zeros((self.Naxis, n), dtype=x.dtype, device=x.device)
        
        # Be careful with boundaries!
        # i=0: Y_0
        # i=1: -2 Y_0 + Y_1
        # i=2: Y_0 - 2 Y_1 + Y_2
        # ...
        dt2 = self.dt ** 2
        if n_2 > 0:
            out[:, 0] = x_mat[:, 0] / dt2
            if n_2 > 1:
                out[:, 1] = (-2.0 * x_mat[:, 0] + x_mat[:, 1]) / dt2
            else:
                out[:, 1] = (-2.0 * x_mat[:, 0]) / dt2
            
            if n_2 > 2:
                out[:, 2:-2] = (x_mat[:, :-2] - 2.0 * x_mat[:, 1:-1] + x_mat[:, 2:]) / dt2
            
            if n_2 > 1:
                out[:, -2] = (x_mat[:, -2] - 2.0 * x_mat[:, -1]) / dt2
                out[:, -1] = x_mat[:, -1] / dt2
            else:
                out[:, -2] = x_mat[:, 0] / dt2

        return out.reshape(-1)

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Proximal Operator (Soft Thresholding).
        """
        x = x.clone()
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)
        x = torch.where(abs_x > self.tv2_lam, sign_x * (abs_x - self.tv2_lam), torch.zeros_like(x))
        return x

    def check(self, x: torch.Tensor) -> None:
        self.hist_feas.append(1)