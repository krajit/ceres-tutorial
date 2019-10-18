#include "ceres/ceres.h"
//#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor
{
  template <typename T>
  bool operator()(const T *const x, T *residual) const
  {
    residual[0] = (8.0 - x[0]);
    return true;
  }
};

int main(int argc, char **argv)
{
  //google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  const double *p = &initial_x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction *cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, &x);

  // double const *const *parameters = &p;
  // double residuals[1];
  // double jacobian[1][1];

  // double parameters[1] = {0.5};
  // double gradient[1];
  // double cost;

  double **parameters = new double *[1];
  parameters[0] = new double[1];
 
  parameters[0][0] = 0.5;

  double **jacobians = new double *[1];
  jacobians[0] = new double[1];

  double residuals = 0.0;

  cost_function->Evaluate(parameters, &residuals, jacobians);

  std::cout << residuals << std::endl;
  std::cout << jacobians[0][0] << std::endl;

  // // Run the solver!
  // Solver::Options options;
  // options.minimizer_progress_to_stdout = true;
  // Solver::Summary summary;
  // Solve(options, &problem, &summary);

  // std::cout << summary.BriefReport() << "\n";
  // std::cout << "x : " << initial_x
  //           << " -> " << x << "\n";
  return 0;
}
