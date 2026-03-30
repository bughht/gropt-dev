import torch
from .base import Operator


class Op_TV_Freq(Operator):
    def __init__(self, tv_lam: float = 0.0, weight_mod: float = 1.0, n_pad: int = 0):
        super().__init__(name="TV_Freq", weight_mod=weight_mod)
        self.tv_lam = float(tv_lam)
        self.N_pad = int(n_pad)

    def init(self, pdata) -> None:
        self.target = 0.0
        self.tol0 = self.tv_lam
        self.tol = (1.0 - self.cushion) * self.tol0

        if self.N_pad <= 0:
            target = max(int(pdata.N * 4), 1024)
            n_pad = 1
            while n_pad < target:
                n_pad *= 2
            self.N_pad = min(n_pad, 32768)

        # We take the derivative of the real and imaginary parts of the FFT.
        # FFT size is N_pad. Difference size is N_pad - 1. Times 2 for real/imag.
        self.Ax_size = pdata.Naxis * (self.N_pad - 1) * 2

        self.spec_norm2 = 1.0
        self.spec_norm = 1.0

        if self.do_init_weights:
            self.obj_weight = 1.0 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * F * X).
        Zero-pads the waveform, takes the FFT, computes the discrete derivative 
        across frequency bins, and returns the real/imag components.
        """
        n = self.N
        x_mat = x.view(self.Naxis, n)
        out = torch.zeros((self.Naxis, self.N_pad), dtype=x.dtype, device=x.device)
        out[:, :n] = x_mat

        # FFT scaling: output is sum(x * exp(...))
        xf = torch.fft.fft(out)
        
        # Calculate discrete derivative in the frequency domain
        diff_xf = torch.diff(xf, dim=1)
        
        # Return as a flattened real tensor
        return torch.cat([diff_xf.real.reshape(-1), diff_xf.imag.reshape(-1)])

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator.
        Exact transpose logic: adjoint of difference, then adjoint of FFT.
        adjoint of FFT matrix without 1/N scaling is N * IFFT.
        """
        n = self.N
        
        x_mat = x.view(2, self.Naxis, self.N_pad - 1)
        diff_xf = torch.complex(x_mat[0], x_mat[1])
        
        # Negative divergence (adjoint of diff)
        out_xf = torch.zeros((self.Naxis, self.N_pad), dtype=diff_xf.dtype, device=x.device)
        out_xf[:, 0] = -diff_xf[:, 0]
        out_xf[:, 1:-1] = diff_xf[:, :-1] - diff_xf[:, 1:]
        out_xf[:, -1] = diff_xf[:, -1]
        
        # Transpose of FFT is strictly N * IFFT applied to the complex spectrum
        out_pad = (torch.fft.ifft(out_xf) * self.N_pad).real
        return out_pad[:, :n].reshape(-1)

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-Linear Proximal Operator (Soft Thresholding).
        Applies L1 regularization to shrink differences in the frequency domain, 
        encouraging a piecewise flat spectrum. (Or L2 if tv_lam is 0).
        """
        x = x.clone()
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)
        x = torch.where(abs_x > self.tv_lam, sign_x * (abs_x - self.tv_lam), torch.zeros_like(x))
        return x

    def check(self, x: torch.Tensor) -> None:
        self.hist_feas.append(1)
