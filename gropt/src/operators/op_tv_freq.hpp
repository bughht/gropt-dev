#ifndef OP_TV_FREQ_H
#define OP_TV_FREQ_H

#include <iostream>
#include <string>
#include <math.h>
#include "Eigen/Dense"

#include "op_main.hpp"
#include "../core/fft_helper.hpp"

namespace Gropt {

class Op_TV_Freq : public Operator
{
    protected:
        double tv_lam = 0.0;
        int n_pad = 0;
        int N_pad = 0;
        std::unique_ptr<FFT_Helper> ffth;

    public:
        Op_TV_Freq(const ProblemData &_pdata, double _tv_lam, double _weight_mod, int _n_pad);
        Op_TV_Freq(const Op_TV_Freq& other);

        virtual std::unique_ptr<Operator> clone(const ProblemData* new_pdata) const override;
        virtual void init();

        virtual void forward(Eigen::VectorXd &X, Eigen::VectorXd &out);
        virtual void transpose(Eigen::VectorXd &X, Eigen::VectorXd &out);
        virtual void prox(Eigen::VectorXd &X);
        virtual void check(Eigen::VectorXd &X);

};

}  // close "namespace Gropt"

#endif
