#include "op_acoustic.hpp"
#include "spdlog/spdlog.h"

namespace Gropt {

Op_Acoustic::Op_Acoustic(const ProblemData &_pdata, std::vector<double> _freqs, std::vector<double> _bws, double _weight_mod) : Operator(_pdata) {
    name = "Acoustic";
    freqs = _freqs;
    bws = _bws;
    weight_mod = _weight_mod;
}

Op_Acoustic::Op_Acoustic(const Op_Acoustic& other) : Operator(other), freqs(other.freqs), bws(other.bws), H(other.H), N_pad(other.N_pad) {
    if (other.ffth) {
        ffth = std::make_unique<FFT_Helper>(other.N_pad);
    }
}

void Op_Acoustic::init() {
    Ax_size = pdata->Naxis * N_pad;
    
    H.setZero(N_pad);
    double df = 1.0 / (N_pad * pdata->dt);
    
    for (int k = 0; k < N_pad; k++) {
        double f = k * df;
        if (k > N_pad / 2) {
            f = (N_pad - k) * df;
        }
        
        bool forbidden = false;
        for (size_t i = 0; i < freqs.size(); i++) {
            if (std::abs(f - freqs[i]) <= bws[i] / 2.0) {
                forbidden = true;
                break;
            }
        }
        if (forbidden) {
            H(k) = 1.0;
        }
    }
    
    ffth = std::make_unique<FFT_Helper>(N_pad);
    
    spec_norm = 1.0;
    spec_norm2 = 1.0;
    
    if (do_init_weights) {
        obj_weight = 1e4; // Similar to b-value/slew constraints
        obj_weight *= weight_mod;
    }

    Operator::init();
}

void Op_Acoustic::forward(Eigen::VectorXd &X, Eigen::VectorXd &out) {
    for (int i_ax = 0; i_ax < pdata->Naxis; i_ax++) {
        Eigen::VectorXd x_pad = Eigen::VectorXd::Zero(N_pad);
        x_pad.head(pdata->N) = X.segment(i_ax * pdata->N, pdata->N);
        Eigen::VectorXd out_pad(N_pad);
        ffth->fft_convolve(x_pad, out_pad, H, false, false);
        out.segment(i_ax * N_pad, N_pad) = out_pad;
    }
}

void Op_Acoustic::transpose(Eigen::VectorXd &X, Eigen::VectorXd &out) {
    // H is real and symmetric, so H^T = H
    for (int i_ax = 0; i_ax < pdata->Naxis; i_ax++) {
        Eigen::VectorXd x_pad = X.segment(i_ax * N_pad, N_pad);
        Eigen::VectorXd out_pad(N_pad);
        ffth->fft_convolve(x_pad, out_pad, H, false, true);
        out.segment(i_ax * pdata->N, pdata->N) = out_pad.head(pdata->N);
    }
}

void Op_Acoustic::prox(Eigen::VectorXd &X) {
    // Exact projection to A X = 0
    X.setZero();
}

void Op_Acoustic::check(Eigen::VectorXd &X) {
    // In strict sense, just check if norm of forbidden bands is near 0
    double err = X.norm();
    int feas = 1;
    if (err > 1e-3) {
        feas = 0;
    }
    hist_feas.push_back(feas);
}

std::unique_ptr<Operator> Op_Acoustic::clone(const ProblemData* new_pdata) const {
    auto ret = std::make_unique<Op_Acoustic>(*this);
    ret->pdata = new_pdata;
    ret->ffth = nullptr;
    return ret;
}

} // namespace
