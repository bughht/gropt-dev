#ifndef OP_CONCOMITANT_H
#define OP_CONCOMITANT_H

/**
 * Constraint on gradient amplitude.  Supports the 'rot_variant' variable
 * to decide if gmax operates per axis or on the gradient magnitude
 *
 * Checks 'set_vals' and forces those values if they are not NaN
 */

#include "Eigen/Dense"
#include <iostream>
#include <math.h>
#include <string>

#include "op_main.hpp"

namespace Gropt {

class Op_Concomitant : public Operator {

  protected:
    int start_idx = 0;

  public:
    Op_Concomitant(const ProblemData &_pdata, int _start_idx, bool _rot_variant, double _weight_mod);
    virtual std::unique_ptr<Operator> clone(const ProblemData* new_pdata) const override;
        virtual void init();

    virtual void forward(Eigen::VectorXd &X, Eigen::VectorXd &out);
    virtual void transpose(Eigen::VectorXd &X, Eigen::VectorXd &out);
    virtual void prox(Eigen::VectorXd &X);
    virtual void check(Eigen::VectorXd &X);
};

} // namespace Gropt

#endif
