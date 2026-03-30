#include "op_tv2.hpp"
#include <cmath>

namespace Gropt {

Op_TV2::Op_TV2(const ProblemData &_pdata, double _tv2_lam, double _weight_mod) : Operator(_pdata) {
    name = "TotalVariation2";
    tv2_lam = _tv2_lam;
    weight_mod = _weight_mod;
}

void Op_TV2::init() {
    target = 0.0;
    tol0 = tv2_lam;
    tol = (1.0 - cushion) * tol0;

    spec_norm2 = 16.0 / (pdata->dt * pdata->dt * pdata->dt * pdata->dt);
    spec_norm = std::sqrt(spec_norm2);

    Ax_size = pdata->Naxis * (pdata->N - 2);

    if (do_init_weights) {
        obj_weight = 1.0 * weight_mod;
    }

    Operator::init();
}

void Op_TV2::forward(Eigen::VectorXd &X, Eigen::VectorXd &out) {
    double dt2 = pdata->dt * pdata->dt;
    for (int i_ax = 0; i_ax < pdata->Naxis; i_ax++) {
        for (int i = 0; i < pdata->N - 2; i++) {
            out(i_ax * (pdata->N - 2) + i) =
                (X(i_ax * pdata->N + i) - 2.0 * X(i_ax * pdata->N + i + 1) + X(i_ax * pdata->N + i + 2)) / dt2;
        }
    }
}

void Op_TV2::transpose(Eigen::VectorXd &X, Eigen::VectorXd &out) {
    out.setZero();
    double dt2 = pdata->dt * pdata->dt;
    int n_2 = pdata->N - 2;
    for (int i_ax = 0; i_ax < pdata->Naxis; i_ax++) {
        if (n_2 > 0) {
            out(i_ax * pdata->N + 0) += X(i_ax * n_2 + 0) / dt2;
            if (n_2 > 1) {
                out(i_ax * pdata->N + 1) += (-2.0 * X(i_ax * n_2 + 0) + X(i_ax * n_2 + 1)) / dt2;
            } else {
                out(i_ax * pdata->N + 1) += -2.0 * X(i_ax * n_2 + 0) / dt2;
            }

            for (int i = 2; i < pdata->N - 2; i++) {
                out(i_ax * pdata->N + i) +=
                    (X(i_ax * n_2 + i - 2) - 2.0 * X(i_ax * n_2 + i - 1) + X(i_ax * n_2 + i)) / dt2;
            }

            if (n_2 > 1) {
                out(i_ax * pdata->N + pdata->N - 2) +=
                    (X(i_ax * n_2 + n_2 - 2) - 2.0 * X(i_ax * n_2 + n_2 - 1)) / dt2;
                out(i_ax * pdata->N + pdata->N - 1) += X(i_ax * n_2 + n_2 - 1) / dt2;
            } else {
                out(i_ax * pdata->N + pdata->N - 2) += X(i_ax * n_2 + 0) / dt2;
                // n_2 == 1 -> pdata->N = 3 -> indices 0, 1, 2
                // We handled out(0), out(1). out(2) handled here.
            }
        }
    }
}

void Op_TV2::prox(Eigen::VectorXd &X) {
    for (int i = 0; i < X.size(); i++) {
        if (X(i) > tv2_lam) {
            X(i) -= tv2_lam;
        } else if (X(i) < -tv2_lam) {
            X(i) += tv2_lam;
        } else {
            X(i) = 0.0;
        }
    }
}

void Op_TV2::check(Eigen::VectorXd &X) {
    hist_feas.push_back(1);
}

std::unique_ptr<Operator> Op_TV2::clone(const ProblemData* new_pdata) const {
    auto ret = std::make_unique<Op_TV2>(*this);
    ret->pdata = new_pdata;
    return ret;
}

} // namespace Gropt