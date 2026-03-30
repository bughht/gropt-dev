from dataclasses import dataclass, field
import torch


@dataclass
class ProblemData:
    dt: float = 0.0
    N: int = 0
    Naxis: int = 1

    inv_vec: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    set_vals: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    fixer: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    X0: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
