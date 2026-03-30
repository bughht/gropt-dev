import math
import torch
from .base import Operator


class Op_BValue(Operator):
    def __init__(self, target: float, tol: float, start_idx0: int, stop_idx0: int,
                 weight_mod: float = 1.0, mode: int = 2, max_scale: float = 1.01):
        super().__init__(name="b-value", weight_mod=weight_mod)
        self.target = float(target)
        self.tol = float(tol)
        self.start_idx0 = int(start_idx0)
        self.stop_idx0 = int(stop_idx0)
        self.mode = int(mode)
        self.max_scale = float(max_scale)

        self.start_idx = self.start_idx0
        self.stop_idx = self.stop_idx0
        self.i_start = 0
        self.i_stop = 0
        self.GAMMA = 0.0
        self.MAT_SCALE = 0.0

    def init(self, pdata) -> None:
        self.target = self.target
        self.tol0 = self.tol
        self.tol = (1.0 - self.cushion) * self.tol0

        self.GAMMA = 267.5221900e6
        self.MAT_SCALE = math.pow((self.GAMMA / 1000.0 * pdata.dt), 2.0) * pdata.dt

        if self.start_idx <= 0:
            self.i_start = 0
        else:
            self.i_start = self.start_idx

        if self.stop_idx <= 0:
            self.i_stop = pdata.N
        else:
            self.i_stop = self.stop_idx

        n_norm = self.i_stop - self.i_start
        self.spec_norm2 = (n_norm * n_norm + n_norm) / 2.0 * self.MAT_SCALE * 0.1175 * 4
        self.spec_norm = math.sqrt(self.spec_norm2)

        self.Ax_size = pdata.Naxis * pdata.N

        if self.do_init_weights:
            self.obj_weight = -1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * X).
        Notice that this only does cumulative sums (integrating gradient to k-space).
        It is purely linear so the CG least-squares solver can handle it perfectly.
        No non-linear squaring happens here.
        """
        n = self.N
        x_mat = x.view(self.Naxis, n)
        inv_mat = self.pdata.inv_vec.view(self.Naxis, n)

        out = torch.zeros_like(x_mat)
        if self.i_stop > self.i_start:
            segment = x_mat[:, self.i_start : self.i_stop] * inv_mat[:, self.i_start : self.i_stop]
            gt = torch.cumsum(segment, dim=1)
            out[:, self.i_start : self.i_stop] = gt * math.sqrt(self.MAT_SCALE)
        return out.reshape(-1)

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_i^T * X).
        Because 'forward' was just a cumsum (lower triangular matrix of 1s),
        the transpose is simply an upper triangular matrix of 1s (reverse cumsum).
        This exact linearity guarantees the Conjugate Gradient solver will converge.
        """
        n = self.N
        x_mat = x.view(self.Naxis, n)
        inv_mat = self.pdata.inv_vec.view(self.Naxis, n)

        out = torch.zeros_like(x_mat)
        if self.i_stop > self.i_start:
            segment = x_mat[:, self.i_start : self.i_stop] * math.sqrt(self.MAT_SCALE)
            rev = torch.flip(segment, dims=[1])
            gt = torch.cumsum(rev, dim=1)
            gt = torch.flip(gt, dims=[1])
            out[:, self.i_start : self.i_stop] = gt * inv_mat[:, self.i_start : self.i_stop]
        return out.reshape(-1)

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Geometry (Proximal Hard Constraint).
        This does the squaring math (calculating the sum-of-squares b-value).
        It strictly forces/scales the 'ghost' k-space trajectory vector to sit perfectly 
        on the exact n-dimensional sphere that gives the target B-value.
        """
        x = x.clone()

        if self.do_equil:
            x = x / self.eq_rows
        x = x * self.spec_norm

        n = self.N
        n_axis = self.Naxis
        for j in range(n_axis):
            seg = x[j * n : (j + 1) * n]
            xnorm = torch.linalg.norm(seg).item()

            if self.mode == 2:
                min_val = math.sqrt(self.target)
                if xnorm < min_val:
                    seg = seg * (min_val / (xnorm + 1.0e-32))
            elif self.mode == 3:
                min_val = math.sqrt(self.target)
                if xnorm < min_val:
                    seg = seg * (self.max_scale * min_val / (xnorm + 1.0e-32))
                else:
                    seg = seg * self.max_scale
            elif self.mode == 1:
                min_val = math.sqrt(self.target - self.tol)
                max_val = math.sqrt(self.target + self.tol)
                if xnorm < min_val:
                    seg = seg * (min_val / (xnorm + 1.0e-32))
                elif xnorm > max_val:
                    seg = seg * (max_val / (xnorm + 1.0e-32))
            else:
                raise ValueError("Unknown BVALUE mode in Op_BValue")

            x[j * n : (j + 1) * n] = seg

        if self.do_equil:
            x = x * self.eq_rows
        x = x / self.spec_norm

        return x

    def check(self, x: torch.Tensor) -> None:
        is_feas = 1

        if self.do_equil:
            x = x / self.eq_rows
        x = x * self.spec_norm

        n = self.N
        n_axis = self.Naxis
        for j in range(n_axis):
            bval_t = torch.sum(x[j * n : (j + 1) * n] ** 2).item()
            if self.mode in (2, 3):
                if bval_t < self.target:
                    is_feas = 0
            elif self.mode == 1:
                if abs(bval_t - self.target) > self.tol0:
                    is_feas = 0
            else:
                raise ValueError("Unknown BVALUE mode in Op_BValue")

        if self.do_equil:
            x = x * self.eq_rows
        x = x / self.spec_norm

        self.hist_feas.append(is_feas)

    def get_bvalue(self, x: torch.Tensor) -> float:
        ax = self.forward_op(x)
        ax = ax * self.spec_norm
        return torch.sum(ax ** 2).item()
