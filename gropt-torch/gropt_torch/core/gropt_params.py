from dataclasses import dataclass
from typing import List
import math
import torch

from .problem_data import ProblemData
from ..operators.base import Operator
from ..operators import Op_Gmax, Op_Smax, Op_Moment, Op_BValue, Op_Acoustic, Op_TV, Op_TV_Freq, Op_TV2, Op_Identity


@dataclass
class SolveResult:
    X: torch.Tensor
    converged: bool = False
    n_iter: int = 0
    n_feval: int = 0
    dt: float = 0.0
    bvalue: float = 0.0


class GroptParams:
    def __init__(self) -> None:
        self.pdata = ProblemData()
        self.all_op: List[Operator] = []
        self.all_obj: List[Operator] = []
        self.Ntot = 0
        self.vec_init_status = -1
        self.op_prep_status = -1
        self.use_cuda = False
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.compile_kernels = False
        self.compile_mode = "default"
        self.compile_backend = "inductor"

    @property
    def dt(self) -> float:
        return self.pdata.dt

    @dt.setter
    def dt(self, val: float) -> None:
        self.pdata.dt = val

    @property
    def N(self) -> int:
        return self.pdata.N

    @N.setter
    def N(self, val: int) -> None:
        self.pdata.N = val

    @property
    def Naxis(self) -> int:
        return self.pdata.Naxis

    @Naxis.setter
    def Naxis(self, val: int) -> None:
        self.pdata.Naxis = val

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self.pdata.inv_vec.numel() > 0:
            self.pdata.inv_vec = self.pdata.inv_vec.to(device=device, dtype=self.dtype)
        if self.pdata.set_vals.numel() > 0:
            self.pdata.set_vals = self.pdata.set_vals.to(device=device, dtype=self.dtype)
        if self.pdata.fixer.numel() > 0:
            self.pdata.fixer = self.pdata.fixer.to(device=device, dtype=self.dtype)
        if self.pdata.X0.numel() > 0:
            self.pdata.X0 = self.pdata.X0.to(device=device, dtype=self.dtype)

    def vec_init_simple(self, N: int = -1, Naxis: int = -1, first_val: float = 0.0, last_val: float = 0.0):
        if N > 0:
            self.N = N
        if Naxis > 0:
            self.Naxis = Naxis
        self.Ntot = self.N * self.Naxis
        self.pdata.inv_vec = torch.ones(self.Ntot, device=self.device, dtype=self.dtype)
        self.pdata.set_vals = torch.full((self.Ntot,), float("nan"), device=self.device, dtype=self.dtype)
        self.pdata.set_vals[0] = first_val
        self.pdata.set_vals[self.N - 1] = last_val
        self.pdata.fixer = torch.ones(self.Ntot, device=self.device, dtype=self.dtype)
        self.pdata.fixer[0] = 0.0
        self.pdata.fixer[self.N - 1] = 0.0
        self.pdata.X0 = torch.full((self.Ntot,), 1e-2, device=self.device, dtype=self.dtype)
        self.pdata.X0[0] = first_val
        self.pdata.X0[self.N - 1] = last_val
        self.vec_init_status = self.N

    # def diff_init(self, dt: float, TE: float, T_90: float, T_180: float, T_readout: float):
    def diff_init(self, dt: float = 400e-6, TE: float = 80e-3, T_90: float = 3e-3,
                  T_180: float = 5e-3, T_readout: float = 16e-3):

        self.dt = dt
        self.Naxis = 1
        self.N = int((TE - T_readout) / dt) + 1
        self.Ntot = self.N * self.Naxis

        ind_inv = int(TE / 2.0 / dt)
        self.pdata.inv_vec = torch.ones(self.N, device=self.device, dtype=self.dtype)
        self.pdata.inv_vec[ind_inv:] = -1.0

        ind_90_end = math.ceil(T_90 / dt)
        ind_180_start = math.floor((TE / 2.0 - T_180 / 2.0) / dt)
        ind_180_end = math.ceil((TE / 2.0 + T_180 / 2.0) / dt)

        self.pdata.set_vals = torch.full((self.N,), float("nan"), device=self.device, dtype=self.dtype)
        self.pdata.set_vals[: ind_90_end + 1] = 0.0
        self.pdata.set_vals[ind_180_start : ind_180_end + 1] = 0.0
        self.pdata.set_vals[0] = 0.0
        self.pdata.set_vals[self.N - 1] = 0.0

        self.pdata.fixer = torch.ones(self.N, device=self.device, dtype=self.dtype)
        self.pdata.fixer[~torch.isnan(self.pdata.set_vals)] = 0.0

        self.pdata.X0 = torch.where(
            torch.isnan(self.pdata.set_vals),
            torch.full((self.N,), 1e-2, device=self.device, dtype=self.dtype),
            self.pdata.set_vals,
        )
        self.pdata.X0 = self.pdata.X0 * self.pdata.inv_vec
        self.vec_init_status = self.N

    def prepare(self):
        self.op_prep_status = self.N
        for op in self.all_op + self.all_obj:
            op.init(self.pdata)

    # Operator adders (stubs, to be implemented in Torch operators)
    def add_gmax(self, gmax: float = 0.03, rot_variant: bool = True, weight_mod: float = 1.0):
        self.op_prep_status = -1
        self.all_op.append(Op_Gmax(gmax=gmax, rot_variant=rot_variant, weight_mod=weight_mod))

    def add_smax(self, smax: float = 80.0, rot_variant: bool = True, weight_mod: float = 1.0):
        self.op_prep_status = -1
        self.all_op.append(Op_Smax(smax=smax, rot_variant=rot_variant, weight_mod=weight_mod))

    def add_moment(self, order: float = 0, target: float = 0.0, tol: float = 1e-6, units: str = "mT*ms/m",
                   axis: int = 0, start_idx: int = -1, stop_idx: int = -1, ref_idx: int = 0, weight_mod: float = 1.0):
        self.op_prep_status = -1
        self.all_op.append(
            Op_Moment(
                order=order,
                target=target,
                tol=tol,
                units=units,
                axis=axis,
                start_idx=start_idx,
                stop_idx=stop_idx,
                ref_idx=ref_idx,
                weight_mod=weight_mod,
            )
        )

    def add_bvalue(self, target: float = 100.0, tol: float = 1.0, start_idx0: int = -1, stop_idx0: int = -1,
                   weight_mod: float = 1.0, mode: int = 2, max_scale: float = 1.01):
        self.op_prep_status = -1
        self.all_op.append(
            Op_BValue(
                target=target,
                tol=tol,
                start_idx0=start_idx0,
                stop_idx0=stop_idx0,
                weight_mod=weight_mod,
                mode=mode,
                max_scale=max_scale,
            )
        )

def add_acoustic(self, freqs, bws, weight_mod: float = 1.0, bw_scale: float = 1.0, n_pad: int = 0):        
        self.op_prep_status = -1
        self.all_op.append(
            Op_Acoustic(freqs=freqs, bws=bws, weight_mod=weight_mod, bw_scale=bw_scale, n_pad=n_pad)      
        )

def add_TV(self, tv_lam: float = 0.0, weight_mod: float = 1.0):
    self.op_prep_status = -1
    self.all_op.append(Op_TV(tv_lam=tv_lam, weight_mod=weight_mod))

def add_TV_Freq(self, tv_lam: float = 0.0, weight_mod: float = 1.0, n_pad: int = 0):
    self.op_prep_status = -1
    self.all_op.append(Op_TV_Freq(tv_lam=tv_lam, weight_mod=weight_mod, n_pad=n_pad))

def add_TV2(self, tv2_lam: float = 0.0, weight_mod: float = 1.0):
    self.op_prep_status = -1
    self.all_op.append(Op_TV2(tv2_lam=tv2_lam, weight_mod=weight_mod))

def add_obj_identity(self, weight_mod: float = 1.0):
    self.op_prep_status = -1
    self.all_obj.append(Op_Identity(weight_mod=weight_mod))
