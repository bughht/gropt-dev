import torch


class Operator:
    def __init__(self, name: str, weight_mod: float = 1.0) -> None:
        self.name = name
        self.weight_mod = weight_mod
        self.Ax_size = 0

        self.pdata = None
        self.N = 0
        self.Naxis = 0
        self.Ntot = 0
        self.dt = 0.0

        self.rot_variant = True
        self.do_init_weights = True

        self.target = 0.0
        self.tol0 = 0.0
        self.tol = 0.0
        self.cushion = 1e-2

        self.spec_norm2 = 1.0
        self.spec_norm = 1.0

        self.obj_weight = 1.0

        self.x_temp = torch.tensor([])
        self.x_temp_obj = torch.tensor([])
        self.Ax_temp = torch.tensor([])

        self.do_equil = False
        self.eq_rows = torch.tensor([])
        self.eq_cols = torch.tensor([])

        self.feas_check = 0.0
        self.r_feas = 0.0
        self.feas_temp = torch.tensor([])

        self.hist_feas = []
        self.hist_r_feas = []

        self._compiled_forward = None
        self._compiled_transpose = None
        self._compiled_prox = None

    def init(self, pdata) -> None:
        self.pdata = pdata
        self.N = pdata.N
        self.Naxis = pdata.Naxis
        self.dt = pdata.dt
        self.Ntot = self.N * self.Naxis

        self.x_temp = torch.zeros(self.Ntot, dtype=pdata.X0.dtype, device=pdata.X0.device)
        self.x_temp_obj = torch.zeros(self.Ntot, dtype=pdata.X0.dtype, device=pdata.X0.device)
        self.Ax_temp = torch.zeros(self.Ax_size, dtype=pdata.X0.dtype, device=pdata.X0.device)

        self.eq_rows = torch.ones(self.Ax_size, dtype=pdata.X0.dtype, device=pdata.X0.device)
        self.eq_cols = torch.ones(self.Ntot, dtype=pdata.X0.dtype, device=pdata.X0.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"forward not implemented for {self.name}")

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"transpose not implemented for {self.name}")

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_equil:
            x_temp = x * self.eq_cols
        else:
            x_temp = x

        out = self._call_forward(x_temp)
        out = out / self.spec_norm
        if self.do_equil:
            out = out * self.eq_rows
        return out

    def transpose_op(self, x: torch.Tensor, apply_fixer: bool = True) -> torch.Tensor:
        if self.do_equil:
            Ax_temp = x * self.eq_rows
        else:
            Ax_temp = x

        out = self._call_transpose(Ax_temp)

        if apply_fixer:
            out = out * self.pdata.fixer
        out = out / self.spec_norm

        if self.do_equil:
            out = out * self.eq_cols
        return out

    def add_Atb(self, b: torch.Tensor, ws) -> torch.Tensor:
        Ax_temp = ws.weight * ws.z0 - ws.y0
        x_temp = self.transpose_op(Ax_temp)
        return b + x_temp

    def add_AtAx(self, x: torch.Tensor, out: torch.Tensor, ws) -> torch.Tensor:
        Ax_temp = self.forward_op(x)
        x_temp = self.transpose_op(Ax_temp)
        return out + ws.weight * x_temp

    def add_obj(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        Ax_temp = self.forward_op(x)
        _ = self.transpose_op(Ax_temp)
        return out + self.obj_weight * Ax_temp

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"prox not implemented for {self.name}")

    def _call_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is not None:
            return self._compiled_forward(x)
        return self.forward(x)

    def _call_transpose(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_transpose is not None:
            return self._compiled_transpose(x)
        return self.transpose(x)

    def _call_prox(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_prox is not None:
            return self._compiled_prox(x)
        return self.prox(x)

    def compile_kernels(self, mode: str = "default", backend: str = "inductor") -> None:
        if not hasattr(torch, "compile"):
            return
        try:
            self._compiled_forward = torch.compile(self.forward, mode=mode, backend=backend)
            self._compiled_transpose = torch.compile(self.transpose, mode=mode, backend=backend)
            self._compiled_prox = torch.compile(self.prox, mode=mode, backend=backend)
        except Exception:
            try:
                self._compiled_forward = torch.compile(self.forward, mode=mode, backend="eager")
                self._compiled_transpose = torch.compile(self.transpose, mode=mode, backend="eager")
                self._compiled_prox = torch.compile(self.prox, mode=mode, backend="eager")
            except Exception:
                self._compiled_forward = None
                self._compiled_transpose = None
                self._compiled_prox = None

    def check(self, x: torch.Tensor) -> None:
        self.feas_check = torch.max(torch.abs(x - self.target)).item()
        self.hist_feas.append(1 if self.feas_check <= self.tol0 else 0)

    def get_feas(self, s: torch.Tensor) -> None:
        feas_temp = self._call_prox(s.clone())
        feas_temp = s - feas_temp
        denom = torch.max(torch.abs(s)).item() + 1.0e-32
        self.r_feas = torch.max(torch.abs(feas_temp)).item() / denom
        self.hist_r_feas.append(self.r_feas)
