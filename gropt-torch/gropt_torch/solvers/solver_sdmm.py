import logging
from dataclasses import dataclass
from typing import List
import torch


logger = logging.getLogger("gropt_torch")


@dataclass
class WorkspaceSolver:
    """
    Base workspace holding the internal variable states for a generic variable-splitting algorithm.
    In the ADMM/SDMM formulation:
      - s: The exact linear output of the forward operator D_i * x.
      - z: The auxiliary "ghost" variable representing the ideal constraint-satisfying state.
      - y: The dual variable (Lagrange multiplier) accumulating the error between 's' and 'z'.
    The 0/1 suffixes represent the (k) and (k+1) iteration states.
    """
    weight: float = 1.0
    gamma: float = 1.5
    do_rw: bool = True
    do_gamma: bool = True
    do_weight: bool = True
    do_scalelim: bool = True

    y0: torch.Tensor = torch.tensor([])
    y1: torch.Tensor = torch.tensor([])
    z0: torch.Tensor = torch.tensor([])
    z1: torch.Tensor = torch.tensor([])
    s0: torch.Tensor = torch.tensor([])
    s1: torch.Tensor = torch.tensor([])

    def init(self, Ax_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.y0 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.y1 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.z0 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.z1 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.s0 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.s1 = torch.zeros(Ax_size, device=device, dtype=dtype)

    def reinit(self, Ax_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.y0.numel() != Ax_size:
            self.init(Ax_size, device, dtype)
        else:
            self.y0.zero_()
            self.y1.zero_()
            self.z0.zero_()
            self.z1.zero_()
            self.s0.zero_()
            self.s1.zero_()

    def prep(self, op, x: torch.Tensor) -> None:
        self.z0 = op.forward_op(x)
        self.z1 = self.z0.clone()


@dataclass
class WorkspaceSDMM(WorkspaceSolver):
    yhat1: torch.Tensor = torch.tensor([])
    dyhat: torch.Tensor = torch.tensor([])
    dy: torch.Tensor = torch.tensor([])
    dhhat: torch.Tensor = torch.tensor([])
    dghat: torch.Tensor = torch.tensor([])
    yhat00: torch.Tensor = torch.tensor([])
    y00: torch.Tensor = torch.tensor([])
    s00: torch.Tensor = torch.tensor([])
    z00: torch.Tensor = torch.tensor([])

    def init(self, Ax_size: int, device: torch.device, dtype: torch.dtype) -> None:
        super().init(Ax_size, device, dtype)
        self.yhat1 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.dyhat = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.dy = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.dhhat = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.dghat = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.yhat00 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.y00 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.s00 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.z00 = torch.zeros(Ax_size, device=device, dtype=dtype)

    def reinit(self, Ax_size: int, device: torch.device, dtype: torch.dtype) -> None:
        super().reinit(Ax_size, device, dtype)
        if self.yhat00.numel() != Ax_size:
            self.yhat00 = torch.zeros(Ax_size, device=device, dtype=dtype)
        if self.y00.numel() != Ax_size:
            self.y00 = torch.zeros(Ax_size, device=device, dtype=dtype)
        self.yhat1.zero_()
        self.dyhat.zero_()
        self.dy.zero_()
        self.dhhat.zero_()
        self.dghat.zero_()
        self.s00.zero_()
        self.z00.zero_()

    def prep(self, op, x: torch.Tensor) -> None:
        super().prep(op, x)
        self.z00 = self.z0.clone()

    def reweight(self, rw_eps: float, e_corr: float, rw_scalelim: float) -> None:
        rho0 = self.weight

        self.yhat1 = self.y0 + rho0 * (self.s1 - self.z1)

        self.dyhat = self.yhat1 - self.yhat00
        self.dy = -(self.y1 - self.y00)
        self.dhhat = self.s1 - self.s00
        self.dghat = -(self.z1 - self.z00)

        norm_dhhat_dyhat = (torch.linalg.norm(self.dhhat) * torch.linalg.norm(self.dyhat)).item()
        dot_dhhat_dhhat = torch.dot(self.dhhat, self.dhhat).item()
        dot_dhhat_dyhat = torch.dot(self.dhhat, self.dyhat).item()

        alpha_corr = 0.0
        if (norm_dhhat_dyhat > rw_eps) and (dot_dhhat_dhhat > rw_eps) and (dot_dhhat_dyhat > rw_eps):
            alpha_corr = dot_dhhat_dyhat / norm_dhhat_dyhat

        norm_dghat_dy = (torch.linalg.norm(self.dghat) * torch.linalg.norm(self.dy)).item()
        dot_dghat_dghat = torch.dot(self.dghat, self.dghat).item()
        dot_dghat_dy = torch.dot(self.dghat, self.dy).item()

        beta_corr = 0.0
        if (norm_dghat_dy > rw_eps) and (dot_dghat_dghat > rw_eps) and (dot_dghat_dy > rw_eps):
            beta_corr = dot_dghat_dy / norm_dghat_dy

        pass_alpha = False
        pass_beta = False

        alpha = 0.0
        if alpha_corr > e_corr:
            pass_alpha = True
            alpha_mg = dot_dhhat_dyhat / dot_dhhat_dhhat
            alpha_sd = torch.dot(self.dyhat, self.dyhat).item() / dot_dhhat_dyhat
            if 2.0 * alpha_mg > alpha_sd:
                alpha = alpha_mg
            else:
                alpha = alpha_sd - 0.5 * alpha_mg

        beta = 0.0
        if beta_corr > e_corr:
            pass_beta = True
            beta_mg = dot_dghat_dy / dot_dghat_dghat
            beta_sd = torch.dot(self.dy, self.dy).item() / dot_dghat_dy
            if 2.0 * beta_mg > beta_sd:
                beta = beta_mg
            else:
                beta = beta_sd - 0.5 * beta_mg

        if pass_alpha and pass_beta:
            step_g1 = (alpha * beta) ** 0.5
            gamma1 = 1.0 + 2.0 * (alpha * beta) ** 0.5 / (alpha + beta)
        elif pass_alpha and not pass_beta:
            step_g1 = alpha
            gamma1 = 1.9
        elif not pass_alpha and pass_beta:
            step_g1 = beta
            gamma1 = 1.1
        else:
            step_g1 = rho0
            gamma1 = 1.5

        if self.do_weight:
            if self.do_scalelim and (step_g1 > rw_scalelim * self.weight):
                self.weight *= rw_scalelim
            elif self.do_scalelim and (rw_scalelim * step_g1 < self.weight):
                self.weight *= 1.0 / rw_scalelim
            else:
                self.weight = step_g1

        if self.do_gamma:
            self.gamma = gamma1

        self.yhat00 = self.yhat1
        self.y00 = self.y1
        self.s00 = self.s1
        self.z00 = self.z1


class IndirectLinearSolver:
    """
    Abstract base class for solving the "Soft Update" (X-update) step of SDMM.
    This module solves the massive unconstrained quadratic least-squares problem:
        min_x  Sum_i [(rho_i / 2) * || D_i*x - z_i + y_i/rho_i ||^2 ] + L2_regularization
    By setting its derivative to zero, this becomes a linear system: (A^T * A) x = A^T * b.
    Because our operators (D_i) are kept strictly linear, we don't need matrix inverses
    or auto-diff; we simply iteratively calculate the left-hand-side and right-hand-side.
    """
    def __init__(self, gparams, n_iter: int, sigma: float, tik_lam: float) -> None:
        self.name = "IndirectLinearSolver"
        self.gparams = gparams
        self.ws: List[WorkspaceSolver] = []
        self.n_iter = n_iter
        self.sigma = sigma
        self.tik_lam = tik_lam
        self.hist_n_iter = [-1]

    def set_workspace(self, ws: List[WorkspaceSolver]) -> None:
        self.ws = list(ws)

    def get_lhs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Left-Hand Side (A^T * A) * x of the normal equations.
        This represents the combined linear curvature of all our constraints together.
        Instead of holding a massive dense matrix in memory, we perform the operation
        dynamically using the forward() and transpose() of each operator.
        """
        out = torch.zeros_like(x)
        if self.tik_lam > 0.0:
            out = out + self.tik_lam * x
        out = out + self.sigma * x

        for op, w in zip(self.gparams.all_op, self.ws):
            # Accumulate D_i^T * rho_i * D_i * x
            out = op.add_AtAx(x, out, w)

        for op in self.gparams.all_obj:
            out = op.add_obj(x, out)

        return out

    def get_rhs(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Right-Hand Side (A^T * b) of the normal equations.
        Here, 'b' is composed of the targeted constraint points (z) minus the dual tension (y).
        This sums up the "tugging" force from all the parallel constraints trying to pull
        the waveform X in their desired direction.
        """
        out = torch.zeros_like(x0)
        out = out + self.sigma * x0
        for op, w in zip(self.gparams.all_op, self.ws):
            # Accumulate D_i^T * rho_i * (z_i - y_i / rho_i)
            out = op.add_Atb(out, w)
        return out

    def solve(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("IndirectLinearSolver.solve is not implemented")


class ILS_CG(IndirectLinearSolver):
    """
    Conjugate Gradient (CG) wrapper to solve the unconstrained, purely linear 
    surrogate problem during the 'Soft Step' of SDMM. It evaluates LHS and RHS
    without forming matrices, allowing it to easily scale to high-resolution MRI arrays.
    """
    def __init__(self, gparams, tol: float, min_iter: int, sigma: float, n_iter: int, tik_lam: float) -> None:
        super().__init__(gparams, n_iter, sigma, tik_lam)
        self.name = "CG"
        self.tol = tol
        self.min_iter = min_iter

    def solve(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Executes the iterative Conjugate Gradient descent for finding 
        the waveform x that compromises between all conflicting constraints perfectly.
        Returns:
            x: An updated waveform balancing the current ADMM penalties.
        """
        x = x0.clone()
        b = self.get_rhs(x0)
        Ax = self.get_lhs(x)

        r = b - Ax
        rnorm0 = torch.linalg.norm(r)
        p = r.clone()
        gamma = torch.dot(r, r)

        ii = 0
        for ii in range(self.n_iter):
            Ap = self.get_lhs(p)
            pAp = torch.dot(p, Ap)
            alpha = gamma / (pAp + 1.0e-32)

            x = x + alpha * p
            r = r - alpha * Ap

            gamma_new = torch.dot(r, r)
            beta = gamma_new / (gamma + 1.0e-32)
            gamma = gamma_new
            p = beta * p + r

            if (torch.sqrt(gamma) <= self.tol * rnorm0) and (ii > self.min_iter):
                break

        self.hist_n_iter.append(ii + 1)
        return x


class SolverGroptSDMM:
    def __init__(self, max_iter: int = 20000):
        self.max_iter = max_iter
        self.max_feval = 12000
        self.log_interval = 20
        self.min_iter = 0
        self.gamma_x = 1.6
        self.extra_iters = 0

        self.ils_tol = 1e-3
        self.ils_max_iter = 10
        self.ils_min_iter = 2
        self.ils_sigma = 1e-4
        self.ils_tik_lam = 1e-4

        self.rw_interval = 8
        self.rw_e_corr = 0.4
        self.rw_eps = 1e-36
        self.rw_scalelim = 1.5

        self.grw_min_infeasible = 20
        self.grw_interval = 20
        self.grw_mod = 2.0

        self.sdmm_ws: List[WorkspaceSDMM] = []
        self.ws: List[WorkspaceSolver] = []
        self.ils_solver = None

        self.hist_cg_iter: List[int] = []
        self.iiter = 0

    def solve(self, gparams):
        if getattr(gparams, "use_cuda", False):
            if torch.cuda.is_available():
                gparams.set_device(torch.device("cuda"))
            else:
                logger.warning("use_cuda=True but CUDA is not available. Falling back to CPU.")

        if gparams.op_prep_status != gparams.N:
            logger.info("Operators do not seem prepared, calling prepare()")
            gparams.prepare()

        if getattr(gparams, "compile_kernels", False):
            mode = getattr(gparams, "compile_mode", "default")
            backend = getattr(gparams, "compile_backend", "inductor")
            for op in gparams.all_op + gparams.all_obj:
                op.compile_kernels(mode=mode, backend=backend)

        device = gparams.pdata.X0.device
        dtype = gparams.pdata.X0.dtype

        self.sdmm_ws = []
        for op in gparams.all_op:
            w = WorkspaceSDMM()
            w.weight = 1.0
            if op.name in ("Slew", "Moment", "b-value", "SAFE", "TotalVariation", "Acoustic"):
                w.weight = 1e4
            w.weight *= op.weight_mod
            w.init(op.Ax_size, device, dtype)
            w.prep(op, gparams.pdata.X0)
            self.sdmm_ws.append(w)

        self.ws = list(self.sdmm_ws)

        X = gparams.pdata.X0.clone()
        Xhat = X.clone()

        self.ils_solver = ILS_CG(gparams, self.ils_tol, self.ils_min_iter, self.ils_sigma,
                                 self.ils_max_iter, self.ils_tik_lam)
        self.ils_solver.set_workspace(self.ws)

        total_feval = 0

        for self.iiter in range(self.max_iter):
            # ====================================================================
            # 1. THE SOFT STEP (X-Update)
            # Find a single composite waveform (Xhat) that acts as a mathematical 
            # compromise between all the divergent constraints and tensions.
            # ====================================================================
            if self.iiter > 0:
                Xhat = self.ils_solver.solve(X)
            else:
                Xhat = X

            if torch.any(torch.abs(Xhat) > 10) or torch.any(torch.isnan(Xhat)):
                logger.error("Large values detected in Xhat at iteration %d. Stopping solver.", self.iiter)
                break

            # ====================================================================
            # 2. THE HARD STEP (Z-Update) & DUAL TENSION UPDATE (Y-Update)
            # See the self.update() method for details on the proximal projections.
            # ====================================================================
            self.update(gparams, Xhat)

            # ====================================================================
            # 3. OVER-RELAXATION
            # Extrapolate the step in the descent direction to improve convergence speed 
            # (Standard ADMM trick when gamma_x > 1.0, usually 1.6 - 1.8)
            # ====================================================================
            X = self.gamma_x * Xhat + (1.0 - self.gamma_x) * X

            self.get_residuals(gparams, X)

            if self.logger(gparams, X) > 0 and (self.iiter > self.min_iter):
                if self.extra_iters > 0:
                    logger.info("First solved at iiter %d, now %d extra iterations", self.iiter, self.extra_iters)
                    self.min_iter = self.iiter + self.extra_iters
                    self.extra_iters = 0
                else:
                    break

            total_feval += self.ils_solver.hist_n_iter[-1]
            if total_feval > self.max_feval:
                logger.info("Maximum function evaluations reached")
                break

        from ..core.gropt_params import SolveResult
        result = SolveResult(X=X, n_iter=self.iiter, dt=gparams.dt)
        self.final_log(gparams, X, result)

        return result

    def update(self, gparams, X: torch.Tensor) -> None:
        """
        Executes the parallel 'Hard Proximal Z-updates' and the 'Dual Y-updates'.
        This loops over constraint operators independently, treating them in parallel.
        """
        for op, w in zip(gparams.all_op, self.sdmm_ws):
            # 1. Forward model calculation check: D_i * X
            # (e.g. Slew rate limits: literally take derivative of X)
            w.s1 = op.forward_op(X)
            
            # 2. Add over-relaxation memory (gamma) and Dual Tensions (y0/weight)
            # This 'pushes' our intermediate value z1 toward a space that historically works
            w.z1 = w.gamma * w.s1 + (1.0 - w.gamma) * w.z0 + w.y0 / w.weight
            
            # 3. Proximal Step (The True Hard Constraint)
            # Take the intermediate variable and rigorously force it to match the physical
            # boundaries via geometric clipping or scaling (non-linear).
            w.z1 = op.prox(w.z1)
            
            # 4. Update the Dual Variables (Accumulate errors)
            # Measure the violation (the difference between what X gave and what the strict Z requires)
            # Add that violation into the running total y1, tightening the 'rubber band'.
            w.y1 = w.y0 + w.weight * (w.gamma * w.s1 + (1.0 - w.gamma) * w.z0 - w.z1)

            # Adaptive Penalty Weighting (SDMM-specific)
            # Dynamically adjusts rho (weight) if primal and dual errors are severely unmatched
            if w.do_rw and (self.iiter > self.rw_interval) and (self.iiter % self.rw_interval == 0):
                w.reweight(self.rw_eps, self.rw_e_corr, self.rw_scalelim)

            # Save state for the next iteration
            w.y0 = w.y1
            w.z0 = w.z1

    def get_residuals(self, gparams, X: torch.Tensor) -> None:
        for op in gparams.all_op:
            ax = op.forward_op(X)
            op.get_feas(ax)
            op.check(ax)

        if self.iiter > 2 * self.grw_min_infeasible and self.iiter % self.grw_interval == 0:
            max_feas = 0.0
            max_index = -1
            for i, op in enumerate(gparams.all_op):
                if sum(op.hist_feas[-self.grw_min_infeasible:]) == 0:
                    if op.hist_r_feas[-1] > max_feas:
                        max_feas = op.hist_r_feas[-1]
                        max_index = i
            if max_index >= 0:
                self.sdmm_ws[max_index].weight *= self.grw_mod

    def logger(self, gparams, X: torch.Tensor) -> int:
        do_print = (self.iiter % self.log_interval == 0)
        all_feasible = 1

        if do_print:
            logger.debug(" ")
            logger.debug("================= Solver Iteration %04d =================", self.iiter)
            if self.ils_solver is not None:
                logger.debug(" Last CG n_iter: %d   ||x|| = %.2e", self.ils_solver.hist_n_iter[-1], torch.linalg.norm(X).item())
            logger.debug("          Name      Feasible   Weight     Gamma     r_feas")
            logger.debug("------------------------------------------------------------------")

        for op, w in zip(gparams.all_op, self.sdmm_ws):
            if do_print:
                logger.debug("    %16s    %d       %.1e    %.1e   %.1e", op.name,
                             op.hist_feas[-1] if op.hist_feas else 0,
                             w.weight, w.gamma,
                             op.hist_r_feas[-1] if op.hist_r_feas else 0.0)
            if op.hist_feas and op.hist_feas[-1] == 0:
                all_feasible = 0

        if self.ils_solver is not None:
            self.hist_cg_iter.append(self.ils_solver.hist_n_iter[-1])

        return all_feasible

    def final_log(self, gparams, X: torch.Tensor, result) -> None:
        result.converged = True
        if self.ils_solver is not None:
            result.n_feval = int(sum(self.ils_solver.hist_n_iter))

        for op in gparams.all_op:
            ax = op.forward_op(X)
            op.check(ax)
            if op.hist_feas and op.hist_feas[-1] == 0:
                result.converged = False

        for op in gparams.all_op:
            if op.name == "b-value":
                if torch.any(torch.isnan(X)):
                    result.bvalue = 0.0
                else:
                    result.bvalue = op.get_bvalue(X)
