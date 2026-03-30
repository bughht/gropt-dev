#include "op_tv_freq.hpp"

namespace Gropt {

Op_TV_Freq::Op_TV_Freq(const ProblemData &_pdata, double _tv_lam, double _weight_mod, int _n_pad)
    : Operator(_pdata), tv_lam(_tv_lam), n_pad(_n_pad) {
    name = "TV_Freq";
    weight_mod = _weight_mod;
}

Op_TV_Freq::Op_TV_Freq(const Op_TV_Freq& other)
    : Operator(other), tv_lam(other.tv_lam), n_pad(other.n_pad), N_pad(other.N_pad) {
    if (other.ffth) {
        ffth = std::make_unique<FFT_Helper>(other.N_pad);
    }
}

void Op_TV_Freq::init() {
    target = 0.0;
    tol0 = tv_lam;
    tol = (1.0 - cushion) * tol0;

    if (n_pad <= 0) {
        int target = std::max(pdata->N * 4, 1024);
        int n_p = 1;
        while (n_p < target) {
            n_p *= 2;
        }
        N_pad = std::min(n_p, 32768);
    } else {
        N_pad = n_pad;
    }

    // Difference size is N_pad - 1. Times 2 for real/imag
    Ax_size = pdata->Naxis * (N_pad - 1) * 2;

    spec_norm2 = 1.0;
    spec_norm = 1.0;

    if (do_init_weights) {
        obj_weight = 1.0 * weight_mod;
    }

    ffth = std::make_unique<FFT_Helper>(N_pad);

    Operator::init();
}

void Op_TV_Freq::forward(Eigen::VectorXd &X, Eigen::VectorXd &out) {
    for (int i_ax = 0; i_ax < pdata->Naxis; i_ax++) {
        Eigen::VectorXd x_pad = Eigen::VectorXd::Zero(N_pad);
        x_pad.head(pdata->N) = X.segment(i_ax * pdata->N, pdata->N);
        
        Eigen::VectorXcd xf;
        ffth->fft(x_pad, xf);

        // Discrete diff
        Eigen::VectorXcd diff_xf(N_pad - 1);
        for (int i = 0; i < N_pad - 1; i++) {
            diff_xf(i) = xf(i + 1) - xf(i);
        }

        // Output format: real features then imag features
        int out_offset = i_ax * (N_pad - 1) * 2;
        out.segment(out_offset, N_pad - 1) = diff_xf.real();
        out.segment(out_offset + (N_pad - 1), N_pad - 1) = diff_xf.imag();
    }
}

void Op_TV_Freq::transpose(Eigen::VectorXd &X, Eigen::VectorXd &out) {
    for (int i_ax = 0; i_ax < pdata->Naxis; i_ax++) {
        int out_offset = i_ax * (N_pad - 1) * 2;
        Eigen::VectorXd diff_xf_real = X.segment(out_offset, N_pad - 1);
        Eigen::VectorXd diff_xf_imag = X.segment(out_offset + (N_pad - 1), N_pad - 1);

        Eigen::VectorXcd diff_xf(N_pad - 1);
        diff_xf.real() = diff_xf_real;
        diff_xf.imag() = diff_xf_imag;

        Eigen::VectorXcd out_xf = Eigen::VectorXcd::Zero(N_pad);
        if (N_pad > 1) {
            out_xf(0) = -diff_xf(0);
            for (int i = 1; i < N_pad - 1; i++) {
                out_xf(i) = diff_xf(i - 1) - diff_xf(i);
            }
            out_xf(N_pad - 1) = diff_xf(N_pad - 2);
        }

        Eigen::VectorXd out_pad(N_pad);
        ffth->ifft(out_xf, out_pad);

        // multiply by N_pad since backward has 1/N scaling
        out_pad *= N_pad;

        out.segment(i_ax * pdata->N, pdata->N) = out_pad.head(pdata->N);
    }
}

void Op_TV_Freq::prox(Eigen::VectorXd &X) {
    if (tv_lam <= 0.0) {
        return;
    }
    
    // Threshold each element independently
    for (int i = 0; i < X.size(); i++) {
        double abs_x = std::abs(X(i));
        if (abs_x > tv_lam) {
            double sign_x = (X(i) > 0) ? 1.0 : -1.0;
            X(i) = sign_x * (abs_x - tv_lam);
        } else {
            X(i) = 0.0;
        }
    }
}

void Op_TV_Freq::check(Eigen::VectorXd &X) {
    hist_feas.push_back(1);
}

std::unique_ptr<Operator> Op_TV_Freq::clone(const ProblemData* new_pdata) const {
    auto ret = std::make_unique<Op_TV_Freq>(*this);
    ret->pdata = new_pdata;
    ret->ffth = nullptr;
    return ret;
}

} // namespace Gropt
