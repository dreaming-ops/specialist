// MEI (unified-sphere + Brown-Conrady incl. k3) reprojection factors with
// analytical Jacobians. Every factor reprojects a point (a feature landmark or
// an AprilTag corner) through one shared lens kernel (MeiCorner) and differs
// only in how the point reaches the body frame; the two finger factors also
// share a RevoluteCorner joint kernel:
//   * MeiFactorCpp            — binary on (body_pose, world_landmark).
//   * MeiWorldPointFactorCpp  — unary on the observing pose; the anchor corner
//                               is a constant, so it pins the world gauge.
//   * MeiFingerFactorCpp      — binary on (theta, body_T_base_tag); a finger tag
//                               on a 1-DoF joint off the base tag.
//   * MeiBaseTagFactorCpp     — unary on body_T_base_tag; the base tag.
//   * MeiCrossTagFactorCpp    — ternary on (observer_pose, owner_pose,
//                               body_T_base_tag); owner's base tag seen by another.
//   * MeiCrossFingerFactorCpp — quaternary, adds the owner's joint angle theta.
// The forward projection mirrors mei_project in specialist.loader.mei_model; the
// analytical Jacobian is native-only, validated against finite differences of it.
// Built by specialist.vi_ba._buildext and loaded by build.py.

#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>

namespace py = pybind11;
using namespace gtsam;

// MEI pixel and d(pixel)/d(p_lens), or false where the cheirality checks fail.
static bool mei_project_jac(const Vector3& p_lens, double fx, double fy,
                            double cx, double cy, double k1, double k2, double k3,
                            double p1, double p2, double xi,
                            const Vector3& obs_bearing, Vector2& pixel_pred,
                            Matrix23& j_mei) {
  if (obs_bearing.squaredNorm() > 0.0 && p_lens.dot(obs_bearing) <= 0.0)
    return false;

  const double norm = p_lens.norm();
  if (norm == 0.0) return false;
  const Vector3 s = p_lens / norm;
  // Pixels need z > -xi, and z > -1/xi for xi > 1, where the image radius folds back.
  const double fold = xi <= 1.0 ? xi : 1.0 / xi;
  if (s(2) <= -fold) return false;
  const double denom = s(2) + xi;

  const double x = s(0) / denom;
  const double y = s(1) / denom;
  const double r2 = x * x + y * y;
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const double radial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 * r2;
  const double x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

  pixel_pred << fx * x_d + cx, fy * y_d + cy;

  const Vector3 e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1);
  const double inv = 1.0 / (norm * denom);
  const Vector3 dx_dp = (e_x - x * e_z - x * xi * s) * inv;
  const Vector3 dy_dp = (e_y - y * e_z - y * xi * s) * inv;
  const Vector3 dr2_dp = 2.0 * x * dx_dp + 2.0 * y * dy_dp;
  const Vector3 dradial_dp = radial_dr2 * dr2_dp;
  const Vector3 dxy_dp = dx_dp * y + x * dy_dp;
  const Vector3 dxd_dp = dx_dp * radial + x * dradial_dp + 2.0 * p1 * dxy_dp +
                         p2 * (dr2_dp + 4.0 * x * dx_dp);
  const Vector3 dyd_dp = dy_dp * radial + y * dradial_dp +
                         p1 * (dr2_dp + 4.0 * y * dy_dp) + 2.0 * p2 * dxy_dp;
  j_mei.row(0) = fx * dxd_dp.transpose();
  j_mei.row(1) = fy * dyd_dp.transpose();
  return true;
}

static Matrix3 skew(const Vector3& v) {
  Matrix3 m;
  m << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  return m;
}

// 2x6 Jacobian w.r.t. the pose a corner is expressed FROM (pose . q), [omega, v].
static Matrix pose_from_jac(const Matrix23& j, const Matrix3& R,
                            const Vector3& q) {
  Matrix H = Matrix::Zero(2, 6);
  H.block(0, 0, 2, 3) = -(j * (R * skew(q)));
  H.block(0, 3, 2, 3) = j * R;
  return H;
}

// 2x6 Jacobian w.r.t. the pose a point is moved TO (pose^-1 . point), p in its frame.
static Matrix pose_to_jac(const Matrix23& j, const Vector3& p) {
  Matrix H = Matrix::Zero(2, 6);
  H.block(0, 0, 2, 3) = j * skew(p);
  H.block(0, 3, 2, 3) = -j;
  return H;
}

using Intrinsics = Eigen::Matrix<double, 10, 1>;  // fx fy cx cy k1 k2 k3 p1 p2 xi

struct MeiCorner {
  double fx, fy, cx, cy, k1, k2, k3, p1, p2, xi;
  Vector2 pixel;
  Vector3 obs_bearing;  // tag direction in the lens frame; zero skips the check
  Matrix3 R_lb;
  Vector3 t_bl;         // lens origin in the body frame

  MeiCorner(const Pose3& body_T_lens, const Intrinsics& k, const Vector2& pixel,
            const Vector3& obs_bearing)
      : fx(k[0]), fy(k[1]), cx(k[2]), cy(k[3]), k1(k[4]), k2(k[5]), k3(k[6]),
        p1(k[7]), p2(k[8]), xi(k[9]), pixel(pixel), obs_bearing(obs_bearing),
        R_lb(body_T_lens.rotation().matrix().transpose()),
        t_bl(body_T_lens.translation()) {}

  bool reproject(const Vector3& p_body, Vector2& residual,
                 Matrix23& j_mei_r_lb) const {
    const Vector3 p_lens = R_lb * (p_body - t_bl);
    Vector2 pixel_pred;
    Matrix23 j_mei;
    if (!mei_project_jac(p_lens, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi,
                         obs_bearing, pixel_pred, j_mei))
      return false;
    residual = pixel_pred - pixel;
    j_mei_r_lb = j_mei * R_lb;
    return true;
  }
};

// The finger factors' revolute joint: q = base_T_joint . Rz(theta) . c and dq/dtheta.
struct RevoluteCorner {
  Matrix3 base_R_joint;
  Vector3 base_t_joint;
  Vector3 c;  // tag corner in the joint frame at theta = 0

  RevoluteCorner(const Pose3& base_T_joint, const Vector3& c)
      : base_R_joint(base_T_joint.rotation().matrix()),
        base_t_joint(base_T_joint.translation()),
        c(c) {}

  void corner(double theta, Vector3& q, Vector3& dq_dtheta) const {
    const double ct = std::cos(theta), st = std::sin(theta);
    const Vector3 p_joint(c(0) * ct - c(1) * st, c(0) * st + c(1) * ct, c(2));
    q = base_R_joint * p_joint + base_t_joint;
    dq_dtheta = base_R_joint * Vector3(-p_joint(1), p_joint(0), 0.0);
  }
};

// p_body = world_T_body^-1 . landmark.
class MeiFactorCpp : public NoiseModelFactorN<Pose3, Point3> {
  using Base = NoiseModelFactorN<Pose3, Point3>;
  MeiCorner lens_;

 public:
  MeiFactorCpp(Key poseKey, Key pointKey, const SharedNoiseModel& model,
               MeiCorner lens)
      : Base(model, poseKey, pointKey), lens_(std::move(lens)) {}

  Vector evaluateError(const Pose3& pose, const Point3& landmark,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2) const override {
    const Matrix3 body_R_world = pose.rotation().matrix().transpose();
    const Vector3 p_body = body_R_world * (landmark - pose.translation());

    Vector2 residual;
    Matrix23 j_body;
    if (!lens_.reproject(p_body, residual, j_body)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_body, p_body);
    if (H2) *H2 = j_body * body_R_world;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiFactorCpp>(*this);
  }
};

// Unary on the observing pose X (world_T_body): a world-fixed point p
// reprojects as p_body = X^-1 . p through the lens. Constant world points
// (anchor tag corners) enter here without becoming graph variables.
class MeiWorldPointFactorCpp : public NoiseModelFactorN<Pose3> {
  using Base = NoiseModelFactorN<Pose3>;
  MeiCorner lens_;
  Vector3 p_world_;

 public:
  MeiWorldPointFactorCpp(Key poseKey, const SharedNoiseModel& model,
                         MeiCorner lens, const Vector3& p_world)
      : Base(model, poseKey), lens_(std::move(lens)), p_world_(p_world) {}

  Vector evaluateError(const Pose3& pose, OptionalMatrixType H1) const override {
    const Matrix3 R_bw = pose.rotation().matrix().transpose();
    const Vector3 p_b = R_bw * (p_world_ - pose.translation());

    Vector2 residual;
    Matrix23 j_mei_r_lb;
    if (!lens_.reproject(p_b, residual, j_mei_r_lb)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_mei_r_lb, p_b);
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiWorldPointFactorCpp>(*this);
  }
};

// Binary on (theta, body_T_base_tag): a finger tag on a 1-DoF revolute joint
// (rotation theta about the joint's +z axis) whose pivot is fixed in the base-tag
// frame (base_T_joint, from the URDF). Camera and tag share the rigid body, so
//   p_body = body_T_base_tag . base_T_joint . Rz(theta) . c
class MeiFingerFactorCpp : public NoiseModelFactorN<double, Pose3> {
  using Base = NoiseModelFactorN<double, Pose3>;
  MeiCorner lens_;
  RevoluteCorner joint_;

 public:
  MeiFingerFactorCpp(Key thetaKey, Key baseTagKey, const SharedNoiseModel& model,
                     MeiCorner lens, RevoluteCorner joint)
      : Base(model, thetaKey, baseTagKey),
        lens_(std::move(lens)),
        joint_(std::move(joint)) {}

  Vector evaluateError(const double& theta, const Pose3& body_T_base_tag,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2) const override {
    Vector3 q, dq_dtheta;
    joint_.corner(theta, q, dq_dtheta);
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 p_body = R_bt * q + body_T_base_tag.translation();

    Vector2 residual;
    Matrix23 j_mei_r_lb;
    if (!lens_.reproject(p_body, residual, j_mei_r_lb)) {
      if (H1) *H1 = Matrix::Zero(2, 1);
      if (H2) *H2 = Matrix::Zero(2, 6);
      return Vector2::Zero();
    }
    if (H1) *H1 = j_mei_r_lb * (R_bt * dq_dtheta);  // joint angle
    if (H2) *H2 = pose_from_jac(j_mei_r_lb, R_bt, q);
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiFingerFactorCpp>(*this);
  }
};

// Unary on body_T_base_tag: the base tag's corner is fixed in the base-tag frame,
// so p_body = body_T_base_tag . c.
class MeiBaseTagFactorCpp : public NoiseModelFactorN<Pose3> {
  using Base = NoiseModelFactorN<Pose3>;
  MeiCorner lens_;
  Vector3 c_;  // tag corner in the base-tag frame

 public:
  MeiBaseTagFactorCpp(Key baseTagKey, const SharedNoiseModel& model,
                      MeiCorner lens, const Vector3& c)
      : Base(model, baseTagKey), lens_(std::move(lens)), c_(c) {}

  Vector evaluateError(const Pose3& body_T_base_tag,
                       OptionalMatrixType H1) const override {
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 p_body = R_bt * c_ + body_T_base_tag.translation();

    Vector2 residual;
    Matrix23 j_mei_r_lb;
    if (!lens_.reproject(p_body, residual, j_mei_r_lb)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_from_jac(j_mei_r_lb, R_bt, c_);
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiBaseTagFactorCpp>(*this);
  }
};

// Ternary on (observer_pose X_O, owner_pose X_W, body_T_base_tag): observer O's
// lens sees owner W's base tag, so
//   p_lensO = lensO_T_bodyO . X_O^-1 . X_W . body_T_base_tag . c
class MeiCrossTagFactorCpp : public NoiseModelFactorN<Pose3, Pose3, Pose3> {
  using Base = NoiseModelFactorN<Pose3, Pose3, Pose3>;
  MeiCorner lens_;
  Vector3 c_;  // base tag corner in the base-tag frame

 public:
  MeiCrossTagFactorCpp(Key observerKey, Key ownerKey, Key baseTagKey,
                       const SharedNoiseModel& model, MeiCorner lens,
                       const Vector3& c)
      : Base(model, observerKey, ownerKey, baseTagKey),
        lens_(std::move(lens)),
        c_(c) {}

  Vector evaluateError(const Pose3& X_O, const Pose3& X_W,
                       const Pose3& body_T_base_tag, OptionalMatrixType H1,
                       OptionalMatrixType H2,
                       OptionalMatrixType H3) const override {
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 m = R_bt * c_ + body_T_base_tag.translation();  // owner body
    const Matrix3 R_W = X_W.rotation().matrix();
    const Vector3 p_world = R_W * m + X_W.translation();
    const Matrix3 R_bwO = X_O.rotation().matrix().transpose();
    const Vector3 p_bodyO = R_bwO * (p_world - X_O.translation());

    Vector2 residual;
    Matrix23 j_mei_r_lb;
    if (!lens_.reproject(p_bodyO, residual, j_mei_r_lb)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 6);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_mei_r_lb, p_bodyO);  // observer pose
    if (H2 || H3) {
      const Matrix23 j_world = j_mei_r_lb * R_bwO;
      if (H2) *H2 = pose_from_jac(j_world, R_W, m);  // owner pose
      if (H3) *H3 = pose_from_jac(j_world * R_W, R_bt, c_);  // base-tag pose
    }
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiCrossTagFactorCpp>(*this);
  }
};

// Quaternary on (X_O, X_W, body_T_base_tag, theta): observer O's lens sees owner
// W's finger tag, which rides W's revolute joint:
//   p_lensO = lensO_T_bodyO . X_O^-1 . X_W . body_T_base_tag . base_T_joint
//             . Rz(theta) . c
class MeiCrossFingerFactorCpp
    : public NoiseModelFactorN<Pose3, Pose3, Pose3, double> {
  using Base = NoiseModelFactorN<Pose3, Pose3, Pose3, double>;
  MeiCorner lens_;
  RevoluteCorner joint_;

 public:
  MeiCrossFingerFactorCpp(Key observerKey, Key ownerKey, Key baseTagKey,
                          Key thetaKey, const SharedNoiseModel& model,
                          MeiCorner lens, RevoluteCorner joint)
      : Base(model, observerKey, ownerKey, baseTagKey, thetaKey),
        lens_(std::move(lens)),
        joint_(std::move(joint)) {}

  Vector evaluateError(const Pose3& X_O, const Pose3& X_W,
                       const Pose3& body_T_base_tag, const double& theta,
                       OptionalMatrixType H1, OptionalMatrixType H2,
                       OptionalMatrixType H3,
                       OptionalMatrixType H4) const override {
    Vector3 q, dq_dtheta;
    joint_.corner(theta, q, dq_dtheta);
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 m = R_bt * q + body_T_base_tag.translation();  // owner body
    const Matrix3 R_W = X_W.rotation().matrix();
    const Vector3 p_world = R_W * m + X_W.translation();
    const Matrix3 R_bwO = X_O.rotation().matrix().transpose();
    const Vector3 p_bodyO = R_bwO * (p_world - X_O.translation());

    Vector2 residual;
    Matrix23 j_mei_r_lb;
    if (!lens_.reproject(p_bodyO, residual, j_mei_r_lb)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 6);
      if (H4) *H4 = Matrix::Zero(2, 1);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_mei_r_lb, p_bodyO);  // observer pose
    const Matrix23 j_world = j_mei_r_lb * R_bwO;
    if (H2) *H2 = pose_from_jac(j_world, R_W, m);  // owner pose
    const Matrix23 j_m = j_world * R_W;  // w.r.t. m (owner-body corner)
    if (H3) *H3 = pose_from_jac(j_m, R_bt, q);  // base-tag pose
    if (H4) *H4 = (j_m * R_bt) * dq_dtheta;  // joint angle
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiCrossFingerFactorCpp>(*this);
  }
};

// The make_* factories return NoiseModelFactor so Python sees its full API.
static NoiseModelFactor::shared_ptr make_mei_factor_cpp(
    size_t poseKey, size_t pointKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens, const Intrinsics& intrinsics, const Vector3& obs_bearing) {
  return std::make_shared<MeiFactorCpp>(
      poseKey, pointKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing));
}

static NoiseModelFactor::shared_ptr make_mei_world_point_factor_cpp(
    size_t poseKey, const Vector2& pixel, const SharedNoiseModel& model,
    const Pose3& body_T_lens, const Vector3& p_world, const Intrinsics& intrinsics, const Vector3& obs_bearing) {
  return std::make_shared<MeiWorldPointFactorCpp>(
      poseKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      p_world);
}

static NoiseModelFactor::shared_ptr make_mei_finger_factor_cpp(
    size_t thetaKey, size_t baseTagKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens,
    const Pose3& base_T_joint, const Vector3& c, const Intrinsics& intrinsics,
    const Vector3& obs_bearing) {
  return std::make_shared<MeiFingerFactorCpp>(
      thetaKey, baseTagKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      RevoluteCorner(base_T_joint, c));
}

static NoiseModelFactor::shared_ptr make_mei_base_tag_factor_cpp(
    size_t baseTagKey, const Vector2& pixel, const SharedNoiseModel& model,
    const Pose3& body_T_lens, const Vector3& c, const Intrinsics& intrinsics,
    const Vector3& obs_bearing) {
  return std::make_shared<MeiBaseTagFactorCpp>(
      baseTagKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      c);
}

static NoiseModelFactor::shared_ptr make_mei_cross_tag_factor_cpp(
    size_t observerKey, size_t ownerKey, size_t baseTagKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& observer_extrinsic,
    const Vector3& c, const Intrinsics& intrinsics,
    const Vector3& obs_bearing) {
  return std::make_shared<MeiCrossTagFactorCpp>(
      observerKey, ownerKey, baseTagKey, model,
      MeiCorner(observer_extrinsic, intrinsics, pixel, obs_bearing),
      c);
}

static NoiseModelFactor::shared_ptr make_mei_cross_finger_factor_cpp(
    size_t observerKey, size_t ownerKey, size_t baseTagKey, size_t thetaKey,
    const Vector2& pixel, const SharedNoiseModel& model,
    const Pose3& observer_extrinsic, const Pose3& base_T_joint, const Vector3& c,
    const Intrinsics& intrinsics, const Vector3& obs_bearing) {
  return std::make_shared<MeiCrossFingerFactorCpp>(
      observerKey, ownerKey, baseTagKey, thetaKey, model,
      MeiCorner(observer_extrinsic, intrinsics, pixel, obs_bearing),
      RevoluteCorner(base_T_joint, c));
}

PYBIND11_MODULE(_mei_factor_cpp, m) {
  m.doc() = "Native MEI reprojection factors with analytical Jacobians.";
  m.def("make_mei_factor_cpp", &make_mei_factor_cpp, py::arg("pose_key"),
        py::arg("point_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("intrinsics"), py::arg("obs_bearing"));
  m.def("make_mei_world_point_factor_cpp", &make_mei_world_point_factor_cpp,
        py::arg("pose_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("p_world"), py::arg("intrinsics"),
        py::arg("obs_bearing"));
  m.def("make_mei_finger_factor_cpp", &make_mei_finger_factor_cpp,
        py::arg("theta_key"), py::arg("base_tag_key"), py::arg("pixel"),
        py::arg("noise"), py::arg("body_T_lens"), py::arg("base_T_joint"),
        py::arg("c"), py::arg("intrinsics"), py::arg("obs_bearing"));
  m.def("make_mei_base_tag_factor_cpp", &make_mei_base_tag_factor_cpp,
        py::arg("base_tag_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("c"), py::arg("intrinsics"),
        py::arg("obs_bearing"));
  m.def("make_mei_cross_tag_factor_cpp", &make_mei_cross_tag_factor_cpp,
        py::arg("observer_key"), py::arg("owner_key"), py::arg("base_tag_key"),
        py::arg("pixel"), py::arg("noise"), py::arg("observer_extrinsic"),
        py::arg("c"), py::arg("intrinsics"), py::arg("obs_bearing"));
  m.def("make_mei_cross_finger_factor_cpp", &make_mei_cross_finger_factor_cpp,
        py::arg("observer_key"), py::arg("owner_key"), py::arg("base_tag_key"),
        py::arg("theta_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("observer_extrinsic"), py::arg("base_T_joint"), py::arg("c"),
        py::arg("intrinsics"), py::arg("obs_bearing"));
}
