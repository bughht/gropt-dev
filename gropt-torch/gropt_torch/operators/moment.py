import math
import torch
from .base import Operator


class Op_Moment(Operator):
    def __init__(self, order: float, target: float, tol: float, units: str, axis: int,
                 start_idx: int, stop_idx: int, ref_idx: int, weight_mod: float = 1.0):
        super().__init__(name="Moment", weight_mod=weight_mod)
        self.order = float(order)
        self.target = float(target)
        self.tol = float(tol)
        self.units = units
        self.axis = int(axis)
        self.start_idx = int(start_idx)
        self.stop_idx = int(stop_idx)
        self.ref_idx = int(ref_idx)

        self._start_idx0 = int(start_idx)
        self._stop_idx0 = int(stop_idx)
        self._ref_idx0 = int(ref_idx)

        self._A = torch.tensor([])

    def init(self, pdata) -> None:
        moment_scale = 1.0
        if self.units == "mT*ms/m":
            moment_scale = 1.0
        elif self.units == "T*s/m":
            moment_scale = 1000.0 * math.pow(1000.0, self.order + 1)
        elif self.units == "rad*s/m":
            moment_scale = 1000.0 * math.pow(1000.0, self.order + 1) / 4.257638544e7
        elif self.units == "s/m":
            moment_scale = 1000.0 * math.pow(1000.0, self.order + 1) / 2.675153194e8
        else:
            raise ValueError(f"Unsupported units for moment constraint: {self.units}")

        self.target = self.target * moment_scale
        self.tol0 = self.tol * moment_scale
        self.tol = (1.0 - self.cushion) * self.tol0

        self.Ax_size = 1

        n = pdata.N
        n_axis = pdata.Naxis
        n_tot = n * n_axis

        if self.start_idx <= 0:
            i_start = self.axis * n
        else:
            i_start = self.start_idx + self.axis * n

        if self.stop_idx <= 0:
            i_stop = (self.axis + 1) * n
        else:
            i_stop = self.stop_idx + self.axis * n

        dtype = pdata.X0.dtype
        device = pdata.X0.device
        A = torch.zeros((1, n_tot), dtype=dtype, device=device)

        self.spec_norm2 = 0.0
        for j in range(i_start, i_stop):
            jj = j - self.axis * n
            val = 1000.0 * 1000.0 * pdata.dt * math.pow((1000.0 * (pdata.dt * (jj - self.ref_idx))), self.order)
            A[0, j] = val * pdata.inv_vec[j]
            self.spec_norm2 += val * val
        self.spec_norm = math.sqrt(self.spec_norm2)

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        self._A = A

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * X).
        Calculates the requested gradient moment (M_0, M_1, M_2...) by multiplying the
        gradient waveform `x` with the pre-calculated polynomial basis array `_A` (i.e. t^n).
        Even though the physics equation has t^n, it is perfectly linear with respect to `x`.
        """
        return self._A.matmul(x)

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_i^T * X).
        Multiplies the scalar constraint/dual error back onto the polynomial basis vector,
        effectively broadcasting the correction force across the entire continuous waveform.
        """
        return self._A.t().matmul(x)

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Geometry (Proximal Hard Constraint).
        If the current moment (`x`) strays exactly outside the `[target - tol, target + tol]`
        window, this instantly hard-clamps the "ghost" scalar variable `z` right onto the boundary.
        """
        x = x.clone()

        if self.do_equil:
            x = x / self.eq_rows
        x = x * self.spec_norm

        lower = self.target - self.tol
        upper = self.target + self.tol
        x = torch.clamp(x, min=lower, max=upper)

        if self.do_equil:
            x = x * self.eq_rows
        x = x / self.spec_norm

        return x

    def check(self, x: torch.Tensor) -> None:
        is_feas = 1

        if self.do_equil:
            x = x / self.eq_rows
        x = x * self.spec_norm

        lower = self.target - self.tol0
        upper = self.target + self.tol0
        for i in range(x.numel()):
            if (x[i] < lower) or (x[i] > upper):
                is_feas = 0

        if self.do_equil:
            x = x * self.eq_rows
        x = x / self.spec_norm

        self.hist_feas.append(is_feas)
