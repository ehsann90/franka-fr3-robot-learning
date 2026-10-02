#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kApprovedAxialForceN = 1.200360;

template <typename Derived>
void print_vector(const std::string& label, const Eigen::MatrixBase<Derived>& value) {
  std::cout << std::left << std::setw(28) << label << "[";
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (index != 0) std::cout << " ";
    std::cout << std::showpos << std::fixed << std::setprecision(8) << value(index);
  }
  std::cout << "]" << std::noshowpos << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " ROBOT_IP" << std::endl;
    return 2;
  }

  try {
    std::cout << "=== DIRECT LIBFRANKA MODEL/JACOBIAN DIAGNOSTIC ===\n";
    std::cout << "mode:                       READ_ONLY\n";
    std::cout << "robot IP:                   " << argv[1] << "\n";
    std::cout << "state access:               readOnce only\n";
    std::cout << "control/motion commands:    DISABLED BY DESIGN\n";
    std::cout << "collision-setting commands: DISABLED BY DESIGN\n";
    std::cout << "approved diagnostic force:  +" << std::fixed << std::setprecision(6)
              << kApprovedAxialForceN << " N along live +Kz\n";
    std::cout << "Press Enter to connect read-only (Ctrl+C cancels): ";
    std::string confirmation;
    std::getline(std::cin, confirmation);

    franka::Robot robot(argv[1]);
    // Match the installed Franky state path: its first readOnce may be stale.
    robot.readOnce();
    const franka::RobotState state = robot.readOnce();
    const franka::Model model = robot.loadModel();

    const auto jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
    const auto coriolis_array = model.coriolis(state);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
    Eigen::Map<const Eigen::Matrix<double, 4, 4>> O_T_EE(state.O_T_EE.data());
    Eigen::Map<const Eigen::Matrix<double, 4, 4>> EE_T_K(state.EE_T_K.data());

    const Eigen::Matrix4d O_T_K = O_T_EE * EE_T_K;
    const Eigen::Vector3d axis_base = O_T_K.topLeftCorner<3, 3>().col(2).normalized();
    Eigen::Matrix<double, 6, 1> wrench_linear_first = Eigen::Matrix<double, 6, 1>::Zero();
    wrench_linear_first.head<3>() = axis_base * kApprovedAxialForceN;
    Eigen::Matrix<double, 6, 1> wrench_angular_first = Eigen::Matrix<double, 6, 1>::Zero();
    wrench_angular_first.tail<3>() = axis_base * kApprovedAxialForceN;
    const Eigen::Matrix<double, 7, 1> task_tau_linear_first =
        jacobian.transpose() * wrench_linear_first;
    const Eigen::Matrix<double, 7, 1> task_tau_angular_first =
        jacobian.transpose() * wrench_angular_first;

    print_vector("live q:", Eigen::Map<const Eigen::Matrix<double, 7, 1>>(state.q.data()));
    print_vector("live +Kz in base:", axis_base);
    print_vector("Jacobian row norms:", jacobian.rowwise().norm());
    print_vector("task tau [F,T]:", task_tau_linear_first);
    print_vector("diagnostic tau [T,F]:", task_tau_angular_first);
    print_vector("coriolis:", coriolis);
    std::cout << std::left << std::setw(28) << "Jacobian max absolute:"
              << std::scientific << std::setprecision(9) << jacobian.cwiseAbs().maxCoeff()
              << std::endl;
    std::cout << std::left << std::setw(28) << "task tau [F,T] norm:"
              << std::scientific << std::setprecision(9) << task_tau_linear_first.norm()
              << std::endl;
    std::cout << "Diagnostic complete. No robot control session or motion command was started."
              << std::endl;
  } catch (const franka::Exception& exception) {
    std::cerr << "libfranka error: " << exception.what() << std::endl;
    return 1;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << std::endl;
    return 1;
  }
  return 0;
}
