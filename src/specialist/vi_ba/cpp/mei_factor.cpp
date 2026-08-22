// MEI reprojection factors with analytical Jacobians, mirroring mei_project.

#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>

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
  Matrix3 lens_R_body;
  Vector3 body_t_lens;

  MeiCorner(const Pose3& body_T_lens, const Intrinsics& k, const Vector2& pixel,
            const Vector3& obs_bearing)
      : fx(k[0]), fy(k[1]), cx(k[2]), cy(k[3]), k1(k[4]), k2(k[5]), k3(k[6]),
        p1(k[7]), p2(k[8]), xi(k[9]), pixel(pixel), obs_bearing(obs_bearing),
        lens_R_body(body_T_lens.rotation().matrix().transpose()),
        body_t_lens(body_T_lens.translation()) {}

  // A seat replaces lens_R_body with seat^T and fills j_seat.
  bool reproject(const Vector3& p_body, Vector2& residual, Matrix23& j_body,
                 const Rot3* seat = nullptr, Matrix23* j_seat = nullptr) const {
    const Matrix3 R = seat ? Matrix3(seat->matrix().transpose()) : lens_R_body;
    const Vector3 p_lens = R * (p_body - body_t_lens);
    Vector2 pixel_pred;
    Matrix23 j_mei;
    if (!mei_project_jac(p_lens, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi,
                         obs_bearing, pixel_pred, j_mei))
      return false;
    residual = pixel_pred - pixel;
    j_body = j_mei * R;
    if (j_seat) *j_seat = j_mei * skew(p_lens);
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

// p_lens = lens_T_body . world_T_body^-1 . world_T_anchor . c; a seat replaces lens_R_body.
class MeiAnchorTagFactorCpp : public NoiseModelFactor {
  MeiCorner lens_;
  Vector3 c_;  // anchor corner in the tag frame

 public:
  MeiAnchorTagFactorCpp(Key poseKey, Key anchorKey, std::optional<Key> seatKey,
                        const SharedNoiseModel& model, MeiCorner lens,
                        const Vector3& c)
      : NoiseModelFactor(model, seatKey ? KeyVector{poseKey, anchorKey, *seatKey}
                                        : KeyVector{poseKey, anchorKey}),
        lens_(std::move(lens)),
        c_(c) {}

  Vector unwhitenedError(const Values& x,
                         OptionalMatrixVecType H = nullptr) const override {
    const Pose3& world_T_body = x.at<Pose3>(keys_[0]);
    const Pose3& world_T_anchor = x.at<Pose3>(keys_[1]);
    const Matrix3 world_R_anchor = world_T_anchor.rotation().matrix();
    const Vector3 p_world = world_R_anchor * c_ + world_T_anchor.translation();
    const Matrix3 body_R_world = world_T_body.rotation().matrix().transpose();
    const Vector3 p_body = body_R_world * (p_world - world_T_body.translation());
    const bool seated = keys_.size() == 3;

    Vector2 residual;
    Matrix23 j_body, j_seat;
    const Rot3 seat = seated ? x.at<Rot3>(keys_[2]) : Rot3();
    const bool active = lens_.reproject(p_body, residual, j_body,
                                        seated ? &seat : nullptr,
                                        seated ? &j_seat : nullptr);
    if (!active) {
      if (H) {
        (*H)[0] = Matrix::Zero(2, 6);
        (*H)[1] = Matrix::Zero(2, 6);
        if (seated) (*H)[2] = Matrix::Zero(2, 3);
      }
      return Vector2::Zero();
    }
    if (H) {
      (*H)[0] = pose_to_jac(j_body, p_body);
      (*H)[1] = pose_from_jac(j_body * body_R_world, world_R_anchor, c_);
      if (seated) (*H)[2] = j_seat;
    }
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiAnchorTagFactorCpp>(*this);
  }
};

// Seated variants: the lens seat is a Rot3 variable with its factory translation.

class MeiSeatedBaseTagFactorCpp : public NoiseModelFactorN<Pose3, Rot3> {
  using Base = NoiseModelFactorN<Pose3, Rot3>;
  MeiCorner lens_;
  Vector3 c_;

 public:
  MeiSeatedBaseTagFactorCpp(Key baseTagKey, Key seatKey,
                            const SharedNoiseModel& model, MeiCorner lens,
                            const Vector3& c)
      : Base(model, baseTagKey, seatKey),
        lens_(std::move(lens)),
        c_(c) {}

  Vector evaluateError(const Pose3& body_T_base_tag, const Rot3& seat,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2) const override {
    const Matrix3 body_R_base_tag = body_T_base_tag.rotation().matrix();
    const Vector3 p_body = body_R_base_tag * c_ + body_T_base_tag.translation();

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject(p_body, residual, j_body, &seat, &j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_from_jac(j_body, body_R_base_tag, c_);
    if (H2) *H2 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedBaseTagFactorCpp>(*this);
  }
};

class MeiSeatedFingerFactorCpp : public NoiseModelFactorN<double, Pose3, Rot3> {
  using Base = NoiseModelFactorN<double, Pose3, Rot3>;
  MeiCorner lens_;
  RevoluteCorner joint_;

 public:
  MeiSeatedFingerFactorCpp(Key thetaKey, Key baseTagKey, Key seatKey,
                           const SharedNoiseModel& model, MeiCorner lens,
                           RevoluteCorner joint)
      : Base(model, thetaKey, baseTagKey, seatKey),
        lens_(std::move(lens)),
        joint_(std::move(joint)) {}

  Vector evaluateError(const double& theta, const Pose3& body_T_base_tag,
                       const Rot3& seat, OptionalMatrixType H1,
                       OptionalMatrixType H2,
                       OptionalMatrixType H3) const override {
    Vector3 q, dq_dtheta;
    joint_.corner(theta, q, dq_dtheta);
    const Matrix3 body_R_base_tag = body_T_base_tag.rotation().matrix();
    const Vector3 p_body = body_R_base_tag * q + body_T_base_tag.translation();

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject(p_body, residual, j_body, &seat, &j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 1);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = j_body * (body_R_base_tag * dq_dtheta);
    if (H2) *H2 = pose_from_jac(j_body, body_R_base_tag, q);
    if (H3) *H3 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedFingerFactorCpp>(*this);
  }
};

// The observer's lens sees a corner of the owner's base tag.
class MeiSeatedCrossTagFactorCpp
    : public NoiseModelFactorN<Pose3, Pose3, Pose3, Rot3> {
  using Base = NoiseModelFactorN<Pose3, Pose3, Pose3, Rot3>;
  MeiCorner lens_;
  Vector3 c_;

 public:
  MeiSeatedCrossTagFactorCpp(Key observerKey, Key ownerKey, Key baseTagKey,
                             Key seatKey, const SharedNoiseModel& model,
                             MeiCorner lens, const Vector3& c)
      : Base(model, observerKey, ownerKey, baseTagKey, seatKey),
        lens_(std::move(lens)),
        c_(c) {}

  Vector evaluateError(const Pose3& world_T_observer, const Pose3& world_T_owner,
                       const Pose3& body_T_base_tag, const Rot3& seat,
                       OptionalMatrixType H1, OptionalMatrixType H2,
                       OptionalMatrixType H3,
                       OptionalMatrixType H4) const override {
    const Matrix3 body_R_base_tag = body_T_base_tag.rotation().matrix();
    const Vector3 p_owner = body_R_base_tag * c_ + body_T_base_tag.translation();
    const Matrix3 world_R_owner = world_T_owner.rotation().matrix();
    const Vector3 p_world = world_R_owner * p_owner + world_T_owner.translation();
    const Matrix3 observer_R_world = world_T_observer.rotation().matrix().transpose();
    const Vector3 p_observer = observer_R_world * (p_world - world_T_observer.translation());

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject(p_observer, residual, j_body, &seat, &j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 6);
      if (H4) *H4 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_body, p_observer);
    if (H2 || H3) {
      const Matrix23 j_world = j_body * observer_R_world;
      if (H2) *H2 = pose_from_jac(j_world, world_R_owner, p_owner);
      if (H3) *H3 = pose_from_jac(j_world * world_R_owner, body_R_base_tag, c_);
    }
    if (H4) *H4 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedCrossTagFactorCpp>(*this);
  }
};

// The seated cross-tag chain through the owner's finger angle theta.
class MeiSeatedCrossFingerFactorCpp
    : public NoiseModelFactorN<Pose3, Pose3, Pose3, double, Rot3> {
  using Base = NoiseModelFactorN<Pose3, Pose3, Pose3, double, Rot3>;
  MeiCorner lens_;
  RevoluteCorner joint_;

 public:
  MeiSeatedCrossFingerFactorCpp(Key observerKey, Key ownerKey, Key baseTagKey,
                                Key thetaKey, Key seatKey,
                                const SharedNoiseModel& model, MeiCorner lens,
                                RevoluteCorner joint)
      : Base(model, observerKey, ownerKey, baseTagKey, thetaKey, seatKey),
        lens_(std::move(lens)),
        joint_(std::move(joint)) {}

  Vector evaluateError(const Pose3& world_T_observer, const Pose3& world_T_owner,
                       const Pose3& body_T_base_tag, const double& theta,
                       const Rot3& seat, OptionalMatrixType H1,
                       OptionalMatrixType H2, OptionalMatrixType H3,
                       OptionalMatrixType H4,
                       OptionalMatrixType H5) const override {
    Vector3 q, dq_dtheta;
    joint_.corner(theta, q, dq_dtheta);
    const Matrix3 body_R_base_tag = body_T_base_tag.rotation().matrix();
    const Vector3 p_owner = body_R_base_tag * q + body_T_base_tag.translation();
    const Matrix3 world_R_owner = world_T_owner.rotation().matrix();
    const Vector3 p_world = world_R_owner * p_owner + world_T_owner.translation();
    const Matrix3 observer_R_world = world_T_observer.rotation().matrix().transpose();
    const Vector3 p_observer = observer_R_world * (p_world - world_T_observer.translation());

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject(p_observer, residual, j_body, &seat, &j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 6);
      if (H4) *H4 = Matrix::Zero(2, 1);
      if (H5) *H5 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    const Matrix23 j_world = j_body * observer_R_world;
    const Matrix23 j_owner = j_world * world_R_owner;
    if (H1) *H1 = pose_to_jac(j_body, p_observer);
    if (H2) *H2 = pose_from_jac(j_world, world_R_owner, p_owner);
    if (H3) *H3 = pose_from_jac(j_owner, body_R_base_tag, q);
    if (H4) *H4 = j_owner * (body_R_base_tag * dq_dtheta);
    if (H5) *H5 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedCrossFingerFactorCpp>(*this);
  }
};

// The make_* factories return NoiseModelFactor so Python sees its full API.
static NoiseModelFactor::shared_ptr make_mei_factor_cpp(
    size_t poseKey, size_t pointKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens, const Intrinsics& intrinsics,
    const Vector3& obs_bearing) {
  return std::make_shared<MeiFactorCpp>(
      poseKey, pointKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing));
}

static NoiseModelFactor::shared_ptr make_mei_seated_base_tag_factor_cpp(
    size_t baseTagKey, size_t seatKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens, const Vector3& c,
    const Intrinsics& intrinsics, const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedBaseTagFactorCpp>(
      baseTagKey, seatKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      c);
}

static NoiseModelFactor::shared_ptr make_mei_seated_finger_factor_cpp(
    size_t thetaKey, size_t baseTagKey, size_t seatKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens,
    const Pose3& base_T_joint, const Vector3& c, const Intrinsics& intrinsics, const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedFingerFactorCpp>(
      thetaKey, baseTagKey, seatKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      RevoluteCorner(base_T_joint, c));
}

static NoiseModelFactor::shared_ptr make_mei_seated_cross_tag_factor_cpp(
    size_t observerKey, size_t ownerKey, size_t baseTagKey, size_t seatKey,
    const Vector2& pixel, const SharedNoiseModel& model,
    const Pose3& body_T_lens, const Vector3& c, const Intrinsics& intrinsics,
    const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedCrossTagFactorCpp>(
      observerKey, ownerKey, baseTagKey, seatKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      c);
}

static NoiseModelFactor::shared_ptr make_mei_seated_cross_finger_factor_cpp(
    size_t observerKey, size_t ownerKey, size_t baseTagKey, size_t thetaKey,
    size_t seatKey, const Vector2& pixel, const SharedNoiseModel& model,
    const Pose3& body_T_lens, const Pose3& base_T_joint, const Vector3& c,
    const Intrinsics& intrinsics, const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedCrossFingerFactorCpp>(
      observerKey, ownerKey, baseTagKey, thetaKey, seatKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      RevoluteCorner(base_T_joint, c));
}

static NoiseModelFactor::shared_ptr make_mei_anchor_tag_factor_cpp(
    size_t poseKey, size_t anchorKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& body_T_lens, const Vector3& c,
    const Intrinsics& intrinsics,
    const Vector3& obs_bearing, std::optional<size_t> seatKey) {
  return std::make_shared<MeiAnchorTagFactorCpp>(
      poseKey, anchorKey, seatKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, obs_bearing),
      c);
}

PYBIND11_MODULE(_mei_factor_cpp, m) {
  m.doc() = "Native MEI reprojection factors with analytical Jacobians.";
  m.def("make_mei_factor_cpp", &make_mei_factor_cpp, py::arg("pose_key"),
        py::arg("point_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("intrinsics"), py::arg("obs_bearing"));
  m.def("make_mei_seated_base_tag_factor_cpp",
        &make_mei_seated_base_tag_factor_cpp, py::arg("base_tag_key"),
        py::arg("seat_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("c"), py::arg("intrinsics"),
        py::arg("obs_bearing"));
  m.def("make_mei_seated_finger_factor_cpp", &make_mei_seated_finger_factor_cpp,
        py::arg("theta_key"), py::arg("base_tag_key"), py::arg("seat_key"),
        py::arg("pixel"), py::arg("noise"), py::arg("body_T_lens"),
        py::arg("base_T_joint"), py::arg("c"), py::arg("intrinsics"),
        py::arg("obs_bearing"));
  m.def("make_mei_seated_cross_tag_factor_cpp",
        &make_mei_seated_cross_tag_factor_cpp, py::arg("observer_key"),
        py::arg("owner_key"), py::arg("base_tag_key"), py::arg("seat_key"),
        py::arg("pixel"), py::arg("noise"), py::arg("body_T_lens"),
        py::arg("c"), py::arg("intrinsics"), py::arg("obs_bearing"));
  m.def("make_mei_seated_cross_finger_factor_cpp",
        &make_mei_seated_cross_finger_factor_cpp, py::arg("observer_key"),
        py::arg("owner_key"), py::arg("base_tag_key"), py::arg("theta_key"),
        py::arg("seat_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("base_T_joint"), py::arg("c"),
        py::arg("intrinsics"), py::arg("obs_bearing"));
  m.def("make_mei_anchor_tag_factor_cpp", &make_mei_anchor_tag_factor_cpp,
        py::arg("pose_key"), py::arg("anchor_key"), py::arg("pixel"),
        py::arg("noise"), py::arg("body_T_lens"), py::arg("c"), py::arg("intrinsics"), py::arg("obs_bearing"),
        py::arg("seat_key") = py::none());
}
