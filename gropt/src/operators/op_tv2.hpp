#ifndef OP_TV2_H
#define OP_TV2_H

#include "Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>

#include "op_main.hpp"

namespace Gropt {

class Op_TV2 : public Operator {
  protected:
    double tv2_lam = 0.0;

  public:
    Op_TV2(const ProblemData &_pdata, double _tv2_lam, double _weight_mod);

    virtual void init();

    virtual void forward(Eigen::VectorXd &X, Eigen::VectorXd &out);
    virtual void transpose(Eigen::VectorXd &X, Eigen::VectorXd &out);
    virtual void prox(Eigen::VectorXd &X);
    virtual void check(Eigen::VectorXd &X);
    virtual std::unique_ptr<Operator> clone(const ProblemData* new_pdata) const override;
};

} // namespace Gropt

#endif