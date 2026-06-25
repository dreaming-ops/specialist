// MEI (unified-sphere + Brown-Conrady incl. k3) reprojection factor on
// (body_pose, world_landmark), with analytical pose and landmark Jacobians.
// The forward projection mirrors mei_project in specialist.loader.mei_model; the
// analytical Jacobian is native-only, validated against finite differences of it.
// Built by specialist.vi_ba._buildext and loaded by build.py.

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <memory>

namespace py = pybind11;
using namespace gtsam;

class MeiFactorCpp : public NoiseModelFactorN<Pose3, Point3> {
  using Base = NoiseModelFactorN<Pose3, Point3>;

  double fx_, fy_, cx_, cy_, k1_, k2_, k3_, p1_, p2_, xi_;
  Vector2 pixel_;
  Matrix3 R_lb_;  // body_T_lens.rotation().transpose()  (lens <- body)
  Vector3 t_bl_;  // body_T_lens.translation()           (body -> lens origin)

 public:
  MeiFactorCpp(Key poseKey, Key pointKey, const Vector2& pixel,
               const SharedNoiseModel& model, const Pose3& body_T_lens, double fx,
               double fy, double cx, double cy, double k1, double k2, double k3,
               double p1, double p2, double xi)
      : Base(model, poseKey, pointKey),
        fx_(fx), fy_(fy), cx_(cx), cy_(cy), k1_(k1), k2_(k2), k3_(k3), p1_(p1),
        p2_(p2), xi_(xi), pixel_(pixel) {
    R_lb_ = body_T_lens.rotation().matrix().transpose();
    t_bl_ = body_T_lens.translation();
  }

  Vector evaluateError(const Pose3& pose, const Point3& landmark,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2) const override {
    const Matrix3 R_wb = pose.rotation().matrix();
    const Vector3 t_wb = pose.translation();
    const Matrix3 R_bw = R_wb.transpose();
    const Vector3 p_b = R_bw * (landmark - t_wb);
    const Vector3 p_lens = R_lb_ * (p_b - t_bl_);

    auto inactive = [&]() -> Vector {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    };

    const double norm = p_lens.norm();
    if (norm == 0.0) return inactive();
    const Vector3 s = p_lens / norm;
    // Pixels need z > -xi, and z > -1/xi for xi > 1, where the image radius folds back.
    const double fold = xi_ <= 1.0 ? xi_ : 1.0 / xi_;
    if (s(2) <= -fold) return inactive();
    const double denom = s(2) + xi_;

    const double x = s(0) / denom;
    const double y = s(1) / denom;
    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2;
    const double radial_dr2 = k1_ + 2.0 * k2_ * r2 + 3.0 * k3_ * r2 * r2;
    const double x_d = x * radial + 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    const double y_d = y * radial + p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;

    Vector2 residual;
    residual << fx_ * x_d + cx_ - pixel_(0), fy_ * y_d + cy_ - pixel_(1);

    if (H1 || H2) {
      const Vector3 e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1);
      const double inv = 1.0 / (norm * denom);
      const Vector3 dx_dp = (e_x - x * e_z - x * xi_ * s) * inv;
      const Vector3 dy_dp = (e_y - y * e_z - y * xi_ * s) * inv;
      const Vector3 dr2_dp = 2.0 * x * dx_dp + 2.0 * y * dy_dp;
      const Vector3 dradial_dp = radial_dr2 * dr2_dp;
      const Vector3 dxy_dp = dx_dp * y + x * dy_dp;
      const Vector3 dxd_dp = dx_dp * radial + x * dradial_dp +
                             2.0 * p1_ * dxy_dp + p2_ * (dr2_dp + 4.0 * x * dx_dp);
      const Vector3 dyd_dp = dy_dp * radial + y * dradial_dp +
                             p1_ * (dr2_dp + 4.0 * y * dy_dp) + 2.0 * p2_ * dxy_dp;
      Matrix23 j_mei;
      j_mei.row(0) = fx_ * dxd_dp.transpose();
      j_mei.row(1) = fy_ * dyd_dp.transpose();
      const Matrix23 j_mei_r_lb = j_mei * R_lb_;
      if (H1) {
        Matrix H = Matrix::Zero(2, 6);
        Matrix3 skew;
        skew << 0.0, -p_b(2), p_b(1), p_b(2), 0.0, -p_b(0), -p_b(1), p_b(0), 0.0;
        H.block(0, 0, 2, 3) = j_mei_r_lb * skew;
        H.block(0, 3, 2, 3) = -j_mei_r_lb;
        *H1 = H;
      }
      if (H2) *H2 = j_mei_r_lb * R_bw;
    }
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiFactorCpp>(*this);
  }
};

using Intrinsics = Eigen::Matrix<double, 10, 1>;  // fx fy cx cy k1 k2 k3 p1 p2 xi

// Returned as NoiseModelFactor (not the NonlinearFactor base) so the Python
// object exposes the full NoiseModelFactor API (unwhitenedError, etc.); it is
// still a NonlinearFactor, so NonlinearFactorGraph.push_back accepts it.
static NoiseModelFactor::shared_ptr make_mei_factor_cpp(
    size_t poseKey, size_t pointKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens,
    const Intrinsics& intrinsics) {
  const Intrinsics& k = intrinsics;
  return std::make_shared<MeiFactorCpp>(poseKey, pointKey, pixel, model,
                                        body_T_lens, k[0], k[1], k[2], k[3], k[4],
                                        k[5], k[6], k[7], k[8], k[9]);
}

PYBIND11_MODULE(_mei_factor_cpp, m) {
  m.doc() = "Native MEI reprojection factor with analytical Jacobians.";
  m.def("make_mei_factor_cpp", &make_mei_factor_cpp, py::arg("pose_key"),
        py::arg("point_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("intrinsics"));
}
