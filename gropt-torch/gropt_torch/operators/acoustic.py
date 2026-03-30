import math
from typing import Sequence
import torch
from .base import Operator


class Op_Acoustic(Operator):
    def __init__(self, freqs: Sequence[float], bws: Sequence[float], weight_mod: float = 1.0,
                 transition_hz: float = 0.0, n_pad: int = 0):
        super().__init__(name="Acoustic", weight_mod=weight_mod)
        self.freqs = [float(f) for f in freqs]
        self.bws = [float(b) for b in bws]
        self.transition_hz = max(0.0, float(transition_hz))

        self.N_pad = int(n_pad)
        self.H = torch.tensor([])

    def init(self, pdata) -> None:
        if self.N_pad <= 0:
            target = max(int(pdata.N * 4), 1024)
            n_pad = 1
            while n_pad < target:
                n_pad *= 2
            self.N_pad = min(n_pad, 32768)

        self.Ax_size = pdata.Naxis * self.N_pad

        df = 1.0 / (self.N_pad * pdata.dt)
        dtype = pdata.X0.dtype
        device = pdata.X0.device
        H = torch.zeros(self.N_pad, dtype=dtype, device=device)

        for k in range(self.N_pad):
            f = k * df
            if k > self.N_pad // 2:
                f = (self.N_pad - k) * df

            h_val = 0.0
            for freq, bw in zip(self.freqs, self.bws):
                dist = abs(f - freq)
                half_bw = bw / 2.0
                if dist <= half_bw:
                    h_val = 1.0
                    break
                if self.transition_hz > 0.0 and dist <= half_bw + self.transition_hz:
                    t = (dist - half_bw) / self.transition_hz
                    taper = 0.5 * (1.0 + math.cos(math.pi * t))
                    if taper > h_val:
                        h_val = taper
            H[k] = h_val

        self.H = H

        self.spec_norm = 1.0
        self.spec_norm2 = 1.0

        if self.do_init_weights:
            self.obj_weight = 1e4 * self.weight_mod

        super().init(pdata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Forward Operator (D_i * X).
        Zero-pads the gradient waveform, performs an FFT, multiplies by a spectral 
        binary/tapered mask (H) isolating the 'dangerous' acoustic resonant frequencies, 
        and performs an IFFT back to the time domain. All linear operations.
        """
        n = self.N
        x_mat = x.view(self.Naxis, n)
        out = torch.zeros((self.Naxis, self.N_pad), dtype=x.dtype, device=x.device)
        out[:, :n] = x_mat

        H = self.H.to(dtype=x.dtype, device=x.device)
        Hc = H.to(dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128)

        xf = torch.fft.fft(out)
        out_pad = torch.fft.ifft(xf * Hc).real
        return out_pad.reshape(-1)

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Linear Transpose Operator (D_i^T * X).
        Because the frequency mask H is purely real and symmetric, the forward matrix
        is symmetric (A^T = A). We just run the FFT mask again and truncate the padding.
        """
        n = self.N
        x_mat = x.view(self.Naxis, self.N_pad)

        H = self.H.to(dtype=x.dtype, device=x.device)
        Hc = H.to(dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128)

        xf = torch.fft.fft(x_mat)
        out_pad = torch.fft.ifft(xf * Hc).real
        return out_pad[:, :n].reshape(-1)

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Non-Linear Geometry (Proximal Hard Constraint).
        The goal of acoustic resonance suppression is perfectly zero energy in the
        target frequency bands. So the ideal "ghost" variable (Z) must have zero
        amplitude coming out of the filter. We hard-clamp the entire vector to 0. 
        """
        return torch.zeros_like(x)

    def check(self, x: torch.Tensor) -> None:
        err = torch.linalg.norm(x).item()
        self.hist_feas.append(0 if err > 1e-3 else 1)
