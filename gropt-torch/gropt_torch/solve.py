from .solvers.solver_sdmm import SolverGroptSDMM


def solve(gparams, max_iter: int = 20000):
    """Solve a gropt-torch problem using the default SDMM solver."""
    solver = SolverGroptSDMM(max_iter=max_iter)
    return solver.solve(gparams)
