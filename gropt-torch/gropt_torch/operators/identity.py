import torch
from .base import Operator


class Op_Identity(Operator):
    def __init__(self, weight_mod: float = 1.0):
        super().__init__(name="Identity", weight_mod=weight_mod)

    def init(self, pdata) -> None:
        self.target = 0.0
        self.spec_norm2 = 1.0
        self.spec_norm = 1.0
        self.Ax_size = pdata.Naxis * pdata.N

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * X).
        Simply returns the waveform exactly as-is. Used as a baseline L2 
        regularization to ensure the problem isn't completely undetermined.
        """
        return x.clone()

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_i^T * X).
        The transpose of an Identity matrix is just the Identity matrix.
        """
        return x.clone()

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Geometry (Proximal Hard Constraint).
        For an L2 energy identity penalty, there are no hard boundaries.
        We just return the current waveform, pulling it purely through the CG step. 
        """
        return x

    def check(self, x: torch.Tensor) -> None:
        self.hist_feas.append(1)
