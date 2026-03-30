import math
import torch
import logging

logger = logging.getLogger("gropt_torch")

def estimate_row_col_norms(gparams, n_reps: int, norm_type: str, row_norms: torch.Tensor, col_norms: torch.Tensor):
    """
    Estimates the maximum inf norms of the rows and columns of the structured matrices 
    constructed by the operators using randomized probing.
    """
    N_rows = sum([op.Ax_size for op in gparams.all_op])
    N_cols = gparams.N * gparams.Naxis
    device = gparams.pdata.X0.device
    dtype = gparams.pdata.X0.dtype

    row_norms.zero_()
    col_norms.zero_()

    for rep in range(n_reps):
        # random vector in {-1, 1}
        z = torch.randint(0, 2, (N_cols,), device=device, dtype=dtype) * 2 - 1
        
        row_start = 0
        for op in gparams.all_op:
            # We call the core forward pass: _call_forward -> div by spec norm
            # But the C++ calls forward_op which applies current eq_rows and eq_cols.
            Ax_temp = op.forward_op(z)
            if norm_type == "L2":
                row_norms[row_start:row_start + op.Ax_size] += torch.abs(Ax_temp) ** 2
            else:
                row_norms[row_start:row_start + op.Ax_size] = torch.maximum(
                    row_norms[row_start:row_start + op.Ax_size], torch.abs(Ax_temp)
                )
            row_start += op.Ax_size

    for rep in range(n_reps):
        Atw = torch.zeros(N_cols, device=device, dtype=dtype)
        for op in gparams.all_op:
            w = torch.randint(0, 2, (op.Ax_size,), device=device, dtype=dtype) * 2 - 1
            x_temp = op.transpose_op(w, apply_fixer=False)
            Atw += x_temp
            
        if norm_type == "L2":
            col_norms += torch.abs(Atw) ** 2
        else:
            col_norms.copy_(torch.maximum(col_norms, torch.abs(Atw)))

    # print(col_norms.max()) # DEBUG

    if norm_type == "L2":
        row_norms.div_(n_reps).sqrt_()
        col_norms.div_(n_reps).sqrt_()


def equilibrate(gparams, n_iter: int = 3, n_reps: int = 10):
    """
    Ruiz Equilibration Protocol:
    Runs over all operators and iteratively finds scaling vectors (eq_rows, eq_cols)
    so that the inf-norm of each operator matrix is approximately 1.0.
    """
    logger.debug(f"Starting equilibration with n_iter={n_iter}, n_reps={n_reps}")
    
    device = gparams.pdata.X0.device
    dtype = gparams.pdata.X0.dtype
    
    N_rows = sum([op.Ax_size for op in gparams.all_op])
    N_cols = gparams.N * gparams.Naxis

    for op in gparams.all_op:
        op.eq_rows = torch.ones(op.Ax_size, device=device, dtype=dtype)
        op.eq_cols = torch.ones(N_cols, device=device, dtype=dtype)
        op.do_equil = True

    row_norms = torch.zeros(N_rows, device=device, dtype=dtype)
    col_norms = torch.zeros(N_cols, device=device, dtype=dtype)

    for iiter in range(n_iter):
        estimate_row_col_norms(gparams, n_reps, "Inf", row_norms, col_norms)

        # Invert and square root
        # Guard against zero-norms (which happen if random probing perfectly 
        # aligns with null-spaces, like adjacent identical elements in differences).
        row_norms = torch.where(row_norms < 1e-12, 1.0, row_norms).rsqrt_()
        col_norms = torch.where(col_norms < 1e-12, 1.0, col_norms).rsqrt_()

        # As per C++ TODO: Fix edges for slew-rate bounds logic
        if col_norms.numel() >= 2:
            col_norms[0] = col_norms[1]
            col_norms[-1] = col_norms[-2]

        row_start = 0
        for op in gparams.all_op:
            op.eq_rows *= row_norms[row_start:row_start + op.Ax_size]
            op.eq_cols *= col_norms
            row_start += op.Ax_size

    logger.debug("Equilibration finished.")
