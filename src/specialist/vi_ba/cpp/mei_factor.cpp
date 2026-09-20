// Native MEI reprojection factors, threaded factor-graph passes and feature-tracking kernels.

#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    const SharedNoiseModel& model, const Pose3& body_T_lens, const Intrinsics& intrinsics) {
  return std::make_shared<MeiFactorCpp>(
      poseKey, pointKey, model,
      MeiCorner(body_T_lens, intrinsics, pixel, Vector3::Zero()));
}

// make_mei_factor_cpp per row, pushed onto `graph` in row order.
static void add_mei_factors_cpp(
    gtsam::NonlinearFactorGraph& graph,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pose_keys,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> point_keys,
    py::array_t<double, py::array::c_style | py::array::forcecast> pixels,
    const SharedNoiseModel& model, const Pose3& body_T_lens, const Intrinsics& intrinsics) {
  const py::ssize_t n = pose_keys.shape(0);
  if (point_keys.shape(0) != n || pixels.ndim() != 2 || pixels.shape(0) != n ||
      pixels.shape(1) != 2)
    throw std::invalid_argument(
        "pose_keys, point_keys and pixels (n, 2) must have one row per factor");
  auto pk = pose_keys.unchecked<1>();
  auto lk = point_keys.unchecked<1>();
  auto px = pixels.unchecked<2>();
  graph.reserve(graph.size() + static_cast<size_t>(n));
  for (py::ssize_t i = 0; i < n; ++i) {
    const Vector2 pixel(px(i, 0), px(i, 1));
    graph.push_back(std::make_shared<MeiFactorCpp>(
        static_cast<size_t>(pk(i)), static_cast<size_t>(lk(i)), model,
        MeiCorner(body_T_lens, intrinsics, pixel, Vector3::Zero())));
  }
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

// work(t) for every t in [0, n_threads), index 0 on the calling thread.
static void run_threads(int n_threads, const std::function<void(int)>& work) {
  std::vector<std::thread> threads;
  for (int t = 1; t < n_threads; ++t) threads.emplace_back(work, t);
  work(0);
  for (auto& th : threads) th.join();
}

static int graph_threads(int requested, size_t n) {
  const int by_size = static_cast<int>(n / 1000) + 1;
  return std::max(1, std::min(requested, by_size));
}

// Every factor's whitened block in graph order, plus (row, col) when `with_pattern`.
static py::tuple linearize_raveled(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> keys,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets,
    int n_threads, bool with_pattern) {
  const size_t n = graph.size();
  std::vector<std::pair<gtsam::Matrix, gtsam::Vector>> blocks(n);
  std::vector<gtsam::KeyVector> block_keys(n);
  std::unordered_map<gtsam::Key, std::pair<int64_t, int64_t>> columns;
  if (with_pattern) {
    if (offsets.ndim() != 2 || offsets.shape(1) != 2 ||
        offsets.shape(0) != keys.shape(0))
      throw std::invalid_argument("offsets must be (len(keys), 2)");
    auto k = keys.unchecked<1>();
    auto o = offsets.unchecked<2>();
    for (py::ssize_t i = 0; i < k.shape(0); ++i)
      columns[static_cast<gtsam::Key>(k(i))] = {o(i, 0), o(i, 1)};
  }
  std::vector<size_t> val_offset(n + 1, 0), row_offset(n + 1, 0);
  py::array_t<double> vals, rhs;
  py::array_t<int64_t> rows, cols;
  {
    py::gil_scoped_release release;
    const int t_count = graph_threads(n_threads, n);
    run_threads(t_count, [&](int t) {
      const size_t begin = n * t / t_count, end = n * (t + 1) / t_count;
      for (size_t i = begin; i < end; ++i) {
        const auto& factor = graph[i];
        if (!factor) continue;
        const auto linear = factor->linearize(values);
        if (!linear) continue;
        blocks[i] = linear->jacobian();
        if (with_pattern) block_keys[i] = linear->keys();
      }
    });
    const char* fault = nullptr;
    for (size_t i = 0; i < n; ++i) {
      val_offset[i + 1] = val_offset[i] + static_cast<size_t>(blocks[i].first.size());
      row_offset[i + 1] = row_offset[i] + static_cast<size_t>(blocks[i].second.size());
      if (!with_pattern || blocks[i].first.size() == 0) continue;
      int64_t width = 0;
      for (const gtsam::Key key : block_keys[i]) {
        const auto found = columns.find(key);
        if (found == columns.end()) fault = "a factor key is not among keys";
        else width += found->second.second;
      }
      if (width != blocks[i].first.cols())
        fault = "a factor's columns do not match its keys";
    }
    if (fault != nullptr) throw std::invalid_argument(fault);
    py::gil_scoped_acquire acquire;
    vals = py::array_t<double>(static_cast<py::ssize_t>(val_offset[n]));
    rhs = py::array_t<double>(static_cast<py::ssize_t>(row_offset[n]));
    rows = py::array_t<int64_t>(static_cast<py::ssize_t>(with_pattern ? val_offset[n] : 0));
    cols = py::array_t<int64_t>(static_cast<py::ssize_t>(with_pattern ? val_offset[n] : 0));
  }
  double* v_out = vals.mutable_data();
  double* r_out = rhs.mutable_data();
  int64_t* row_out = with_pattern ? rows.mutable_data() : nullptr;
  int64_t* col_out = with_pattern ? cols.mutable_data() : nullptr;
  {
    py::gil_scoped_release release;
    const int t_count = graph_threads(n_threads, n);
    run_threads(t_count, [&](int t) {
      const size_t begin = n * t / t_count, end = n * (t + 1) / t_count;
      std::vector<int64_t> col_index;
      for (size_t i = begin; i < end; ++i) {
        const gtsam::Matrix& A = blocks[i].first;
        const gtsam::Vector& b = blocks[i].second;
        double* v = v_out + val_offset[i];
        for (Eigen::Index r = 0; r < A.rows(); ++r)
          for (Eigen::Index c = 0; c < A.cols(); ++c) *v++ = A(r, c);
        double* rr = r_out + row_offset[i];
        for (Eigen::Index r = 0; r < b.size(); ++r) *rr++ = b(r);
        if (!with_pattern || A.size() == 0) continue;
        col_index.clear();
        for (const gtsam::Key key : block_keys[i]) {
          const auto& column = columns.find(key)->second;
          for (int64_t c = 0; c < column.second; ++c)
            col_index.push_back(column.first + c);
        }
        int64_t* ro = row_out + val_offset[i];
        int64_t* co = col_out + val_offset[i];
        const int64_t row0 = static_cast<int64_t>(row_offset[i]);
        for (Eigen::Index r = 0; r < A.rows(); ++r)
          for (Eigen::Index c = 0; c < A.cols(); ++c) {
            *ro++ = row0 + r;
            *co++ = col_index[static_cast<size_t>(c)];
          }
      }
    });
  }
  return py::make_tuple(vals, rhs, rows, cols);
}

// A copy of `graph` without the factors that touch any of `drop_keys`.
static gtsam::NonlinearFactorGraph prune_factors(
    const gtsam::NonlinearFactorGraph& graph,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> drop_keys) {
  std::unordered_set<gtsam::Key> dropped;
  auto d = drop_keys.unchecked<1>();
  for (py::ssize_t i = 0; i < d.shape(0); ++i)
    dropped.insert(static_cast<gtsam::Key>(d(i)));
  gtsam::NonlinearFactorGraph kept;
  kept.reserve(graph.size());
  for (const auto& factor : graph) {
    if (!factor) continue;
    const auto& keys = factor->keys();
    if (std::none_of(keys.begin(), keys.end(),
                     [&](gtsam::Key k) { return dropped.count(k) > 0; }))
      kept.push_back(factor);
  }
  return kept;
}

// graph.error(values) on `n_threads`, summed in factor order.
static double graph_error(const gtsam::NonlinearFactorGraph& graph,
                          const gtsam::Values& values, int n_threads) {
  const size_t n = graph.size();
  std::vector<double> errors(n, 0.0);
  py::gil_scoped_release release;
  const int t_count = graph_threads(n_threads, n);
  run_threads(t_count, [&](int t) {
    const size_t begin = n * t / t_count, end = n * (t + 1) / t_count;
    for (size_t i = begin; i < end; ++i)
      if (graph[i]) errors[i] = graph[i]->error(values);
  });
  double total = 0.0;
  for (size_t i = 0; i < n; ++i)
    if (graph[i]) total += errors[i];
  return total;
}

// A flat tangent vector as per-key VectorValues, key i at offsets[i].
static gtsam::VectorValues scatter_delta(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> keys,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> delta) {
  if (offsets.ndim() != 2 || offsets.shape(1) != 2 ||
      offsets.shape(0) != keys.shape(0))
    throw std::invalid_argument("offsets must be (len(keys), 2)");
  auto k = keys.unchecked<1>();
  auto o = offsets.unchecked<2>();
  const double* d = delta.data();
  const int64_t n_delta = static_cast<int64_t>(delta.shape(0));
  gtsam::VectorValues out;
  for (py::ssize_t i = 0; i < k.shape(0); ++i) {
    const int64_t off = o(i, 0), dim = o(i, 1);
    if (off < 0 || dim < 0 || off + dim > n_delta)
      throw std::invalid_argument("an offset runs past delta");
    out.insert(static_cast<gtsam::Key>(k(i)),
               gtsam::Vector(Eigen::Map<const gtsam::Vector>(d + off, dim)));
  }
  return out;
}

// cv2::goodFeaturesToTrack's greedy min-distance pass over ordered candidates.
static std::vector<int64_t> greedy_corners(const int32_t* x, const int32_t* y,
                                           int n, int width, int height,
                                           double min_distance, int max_corners) {
  const int cell = static_cast<int>(std::lround(min_distance));
  const int grid_w = (width + cell - 1) / cell;
  const int grid_h = (height + cell - 1) / cell;
  std::vector<std::vector<std::pair<float, float>>> grid(
      static_cast<size_t>(grid_w) * grid_h);
  const double min_sq = min_distance * min_distance;
  std::vector<int64_t> kept;
  kept.reserve(static_cast<size_t>(std::max(0, std::min(n, max_corners))));
  for (int i = 0; i < n && static_cast<int>(kept.size()) < max_corners; ++i) {
    const float px = static_cast<float>(x[i]);
    const float py_ = static_cast<float>(y[i]);
    const int cx = static_cast<int>(px) / cell;
    const int cy = static_cast<int>(py_) / cell;
    const int x1 = std::max(cx - 1, 0), y1 = std::max(cy - 1, 0);
    const int x2 = std::min(cx + 1, grid_w - 1), y2 = std::min(cy + 1, grid_h - 1);
    bool good = true;
    for (int yy = y1; yy <= y2 && good; ++yy)
      for (int xx = x1; xx <= x2 && good; ++xx)
        for (const auto& q : grid[static_cast<size_t>(yy) * grid_w + xx]) {
          const float dx = px - q.first, dy = py_ - q.second;
          if (dx * dx + dy * dy < min_sq) {
            good = false;
            break;
          }
        }
    if (good) {
      grid[static_cast<size_t>(cy) * grid_w + cx].emplace_back(px, py_);
      kept.push_back(i);
    }
  }
  return kept;
}

static int scan_threads(int rows) {
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  return std::max(1, std::min({8, cores, rows / 64}));
}

// cv2::goodFeaturesToTrack on a computed corner response; (N, 2) pixels.
static py::array_t<double> detect_corners(
    py::array_t<float, py::array::c_style | py::array::forcecast> response,
    py::object mask_obj, double quality_level, double min_distance,
    int max_corners) {
  if (response.ndim() != 2) throw std::invalid_argument("response must be 2-D");
  const int h = static_cast<int>(response.shape(0));
  const int w = static_cast<int>(response.shape(1));
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask_arr;
  const uint8_t* mask = nullptr;
  if (!mask_obj.is_none()) {
    mask_arr = mask_obj.cast<
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    if (mask_arr.ndim() != 2 || mask_arr.shape(0) != h || mask_arr.shape(1) != w)
      throw std::invalid_argument("mask must match the response's shape");
    mask = mask_arr.data();
  }
  const float* r = response.data();
  std::vector<int32_t> xs, ys;
  std::vector<int64_t> kept;
  {
    py::gil_scoped_release release;
    const size_t n_px = static_cast<size_t>(h) * w;
    const int n_threads = scan_threads(h);
    std::vector<double> part_max(n_threads, -std::numeric_limits<double>::infinity());
    std::vector<char> part_any(n_threads, 0);
    run_threads(n_threads, [&](int t) {
      const size_t begin = n_px * t / n_threads, end = n_px * (t + 1) / n_threads;
      double best = -std::numeric_limits<double>::infinity();
      bool any = false;
      for (size_t i = begin; i < end; ++i) {
        if (mask != nullptr && mask[i] == 0) continue;
        any = true;
        if (r[i] > best) best = r[i];
      }
      part_max[t] = best;
      part_any[t] = any;
    });
    double max_value = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (int t = 0; t < n_threads; ++t) {
      if (!part_any[t]) continue;
      any = true;
      if (part_max[t] > max_value) max_value = part_max[t];
    }
    if (!any || max_corners <= 0 || h < 3 || w < 3)
      return py::array_t<double>(std::vector<py::ssize_t>{0, 2});
    const float thr = static_cast<float>(max_value * quality_level);
    std::vector<std::vector<int32_t>> part_x(n_threads), part_y(n_threads);
    std::vector<std::vector<float>> part_v(n_threads);
    auto scan = [&](int t) {
      const int y_begin = std::max(1, 1 + (h - 2) * t / n_threads);
      const int y_end = std::max(1, 1 + (h - 2) * (t + 1) / n_threads);
      for (int y = y_begin; y < y_end; ++y) {
        const float* row = r + static_cast<size_t>(y) * w;
        const float* up = row - w;
        const float* down = row + w;
        const uint8_t* mrow =
            mask == nullptr ? nullptr : mask + static_cast<size_t>(y) * w;
        for (int x = 1; x < w - 1; ++x) {
          const float v = row[x];
          if (!(v > thr)) continue;
          if (mrow != nullptr && mrow[x] == 0) continue;
          if (up[x - 1] > v || up[x] > v || up[x + 1] > v || row[x - 1] > v ||
              row[x + 1] > v || down[x - 1] > v || down[x] > v || down[x + 1] > v)
            continue;
          part_x[t].push_back(x);
          part_y[t].push_back(y);
          part_v[t].push_back(v);
        }
      }
    };
    run_threads(n_threads, scan);
    std::vector<float> vs;
    for (int t = 0; t < n_threads; ++t) {
      xs.insert(xs.end(), part_x[t].begin(), part_x[t].end());
      ys.insert(ys.end(), part_y[t].begin(), part_y[t].end());
      vs.insert(vs.end(), part_v[t].begin(), part_v[t].end());
    }
    // The ranking (response desc, raster) is produced a doubling leading run at a time.
    const size_t n_cand = xs.size();
    std::vector<int64_t> order(n_cand);
    for (size_t i = 0; i < n_cand; ++i) order[i] = static_cast<int64_t>(i);
    const auto ahead = [&vs](int64_t a, int64_t b) {
      return vs[a] > vs[b] || (vs[a] == vs[b] && a < b);
    };
    std::vector<int32_t> sx, sy;
    size_t run = std::min(n_cand, static_cast<size_t>(16) * max_corners);
    while (true) {
      if (run < n_cand)
        std::nth_element(order.begin(), order.begin() + run, order.end(), ahead);
      std::sort(order.begin(), order.begin() + run, ahead);
      sx.resize(run);
      sy.resize(run);
      for (size_t i = 0; i < run; ++i) {
        sx[i] = xs[static_cast<size_t>(order[i])];
        sy[i] = ys[static_cast<size_t>(order[i])];
      }
      kept = greedy_corners(sx.data(), sy.data(), static_cast<int>(run), w, h,
                            min_distance, max_corners);
      if (run == n_cand || static_cast<int>(kept.size()) >= max_corners) break;
      run = std::min(n_cand, run * 2);
    }
    xs.swap(sx);
    ys.swap(sy);
  }
  py::array_t<double> out(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(kept.size()), 2});
  auto o = out.mutable_unchecked<2>();
  for (size_t i = 0; i < kept.size(); ++i) {
    o(static_cast<py::ssize_t>(i), 0) = xs[static_cast<size_t>(kept[i])];
    o(static_cast<py::ssize_t>(i), 1) = ys[static_cast<size_t>(kept[i])];
  }
  return out;
}

// 255 everywhere but on the (dy, dx) disc stamped around every center.
static py::array_t<uint8_t> free_mask(
    int height, int width,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> centers,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> dy,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> dx) {
  if (centers.ndim() != 2 || centers.shape(1) != 2 || dy.shape(0) != dx.shape(0))
    throw std::invalid_argument("centers must be (n, 2) and dy, dx alike");
  py::array_t<uint8_t> out(std::vector<py::ssize_t>{height, width});
  uint8_t* m = out.mutable_data();
  auto c = centers.unchecked<2>();
  const int64_t* oy = dy.data();
  const int64_t* ox = dx.data();
  const py::ssize_t n_off = dy.shape(0);
  {
    py::gil_scoped_release release;
    std::fill(m, m + static_cast<size_t>(height) * width, 255);
    for (py::ssize_t i = 0; i < c.shape(0); ++i) {
      const int64_t cx = c(i, 0), cy = c(i, 1);
      for (py::ssize_t k = 0; k < n_off; ++k) {
        const int64_t y = cy + oy[k], x = cx + ox[k];
        if (y < 0 || y >= height || x < 0 || x >= width) continue;
        m[static_cast<size_t>(y) * width + x] = 0;
      }
    }
  }
  return out;
}

// Bilinear flow sample at a subpixel position, lifted back through `scale`.
static void sample_flow(const float* flow, int fh, int fw, double scale,
                        double x_in, double y_in, double& dx, double& dy) {
  const double x_max = static_cast<double>(fw) - 1.001;
  const double y_max = static_cast<double>(fh) - 1.001;
  const double x = std::min(std::max((x_in + 0.5) * scale - 0.5, 0.0), x_max);
  const double y = std::min(std::max((y_in + 0.5) * scale - 0.5, 0.0), y_max);
  const int64_t x0 = static_cast<int64_t>(x);
  const int64_t y0 = static_cast<int64_t>(y);
  const double fx = x - static_cast<double>(x0);
  const double fy = y - static_cast<double>(y0);
  const float* f00 = flow + (y0 * fw + x0) * 2;
  const float* f01 = f00 + 2;
  const float* f10 = f00 + static_cast<int64_t>(fw) * 2;
  const float* f11 = f10 + 2;
  double s[2];
  for (int c = 0; c < 2; ++c) {
    s[c] = (1.0 - fx) * (1.0 - fy) * static_cast<double>(f00[c]) +
           fx * (1.0 - fy) * static_cast<double>(f01[c]) +
           (1.0 - fx) * fy * static_cast<double>(f10[c]) +
           fx * fy * static_cast<double>(f11[c]);
    if (scale != 1.0) s[c] = s[c] / scale;
  }
  dx = s[0];
  dy = s[1];
}

// Prior points moved by the forward flow, gated by fb error and the frame border.
static py::tuple propagate_points(
    py::array_t<float, py::array::c_style | py::array::forcecast> flow_fwd,
    py::array_t<float, py::array::c_style | py::array::forcecast> flow_bwd,
    py::array_t<double, py::array::c_style | py::array::forcecast> prev_xy,
    double scale, int height, int width, double fb_threshold) {
  if (flow_fwd.ndim() != 3 || flow_fwd.shape(2) != 2 ||
      flow_bwd.ndim() != 3 || flow_bwd.shape(2) != 2 ||
      flow_fwd.shape(0) != flow_bwd.shape(0) ||
      flow_fwd.shape(1) != flow_bwd.shape(1))
    throw std::invalid_argument("flows must be (H, W, 2) and alike");
  if (prev_xy.ndim() != 2 || prev_xy.shape(1) != 2)
    throw std::invalid_argument("prev_xy must be (N, 2)");
  const int fh = static_cast<int>(flow_fwd.shape(0));
  const int fw = static_cast<int>(flow_fwd.shape(1));
  const py::ssize_t n = prev_xy.shape(0);
  py::array_t<double> next(std::vector<py::ssize_t>{n, 2});
  py::array_t<bool> ok(n);
  const float* fwd = flow_fwd.data();
  const float* bwd = flow_bwd.data();
  const double* prev = prev_xy.data();
  double* nxt = next.mutable_data();
  bool* good = ok.mutable_data();
  {
    py::gil_scoped_release release;
    const double x_lim = static_cast<double>(width - 1);
    const double y_lim = static_cast<double>(height - 1);
    for (py::ssize_t i = 0; i < n; ++i) {
      const double px = prev[2 * i], py_ = prev[2 * i + 1];
      if (!(std::isfinite(px) && std::isfinite(py_))) {
        nxt[2 * i] = nxt[2 * i + 1] = std::numeric_limits<double>::quiet_NaN();
        good[i] = false;
        continue;
      }
      double dx, dy;
      sample_flow(fwd, fh, fw, scale, px, py_, dx, dy);
      const double nx = px + dx, ny = py_ + dy;
      double bx, by;
      sample_flow(bwd, fh, fw, scale, nx, ny, bx, by);
      const double ex = (nx + bx) - px, ey = (ny + by) - py_;
      const double fb = std::sqrt(ex * ex + ey * ey);
      nxt[2 * i] = nx;
      nxt[2 * i + 1] = ny;
      good[i] = std::isfinite(fb) && fb < fb_threshold && nx >= 1.0 &&
                nx < x_lim && ny >= 1.0 && ny < y_lim;
    }
  }
  return py::make_tuple(next, ok);
}

PYBIND11_MODULE(_mei_factor_cpp, m) {
  m.doc() = "Native MEI factors, threaded factor-graph passes and feature-tracking kernels.";
  m.def("linearize_raveled", &linearize_raveled, py::arg("graph"),
        py::arg("values"), py::arg("keys"), py::arg("offsets"),
        py::arg("n_threads"), py::arg("with_pattern"));
  m.def("graph_error", &graph_error, py::arg("graph"), py::arg("values"),
        py::arg("n_threads"));
  m.def("prune_factors", &prune_factors, py::arg("graph"), py::arg("drop_keys"));
  m.def("scatter_delta", &scatter_delta, py::arg("keys"), py::arg("offsets"),
        py::arg("delta"));
  m.def("detect_corners", &detect_corners, py::arg("response"), py::arg("mask"),
        py::arg("quality_level"), py::arg("min_distance"),
        py::arg("max_corners"));
  m.def("free_mask", &free_mask, py::arg("height"), py::arg("width"),
        py::arg("centers"), py::arg("dy"), py::arg("dx"));
  m.def("propagate_points", &propagate_points, py::arg("flow_fwd"),
        py::arg("flow_bwd"), py::arg("prev_xy"), py::arg("scale"),
        py::arg("height"), py::arg("width"), py::arg("fb_threshold"));
  m.def("add_mei_factors_cpp", &add_mei_factors_cpp, py::arg("graph"),
        py::arg("pose_keys"), py::arg("point_keys"), py::arg("pixels"),
        py::arg("noise"), py::arg("body_T_lens"), py::arg("intrinsics"));
  m.def("make_mei_factor_cpp", &make_mei_factor_cpp, py::arg("pose_key"),
        py::arg("point_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("body_T_lens"), py::arg("intrinsics"));
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
