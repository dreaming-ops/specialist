/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    config.h
 * @brief   Settings and paths configured with CMake
 * @author  Richard Roberts
 */

// Vendored copy of gtsam's cmake-generated config.h for gtsam 4.3a0, matching
// the options of the official PyPI wheel (quaternions off, EXPMAP on, STL
// allocator, tangent preintegration, boost features on, no TBB). Committed so
// _mei_factor_cpp compiles against the wheel's ABI without running gtsam's full
// cmake configure (which needs Boost component libraries absent on minimal
// Linux installs). See specialist/vi_ba/_buildext.py.
//
// To refresh on a gtsam version bump: clone the matching gtsam source, run
// `cmake -B build` once, copy the generated build/gtsam/config.h here, and
// re-neutralize the two dataset-dir paths below (they otherwise embed an
// absolute build path). Keep the option set identical to the target wheel.

#pragma once

// Library version
#define GTSAM_VERSION_MAJOR 4
#define GTSAM_VERSION_MINOR 3
#define GTSAM_VERSION_PATCH 0
#define GTSAM_VERSION_NUMERIC 40300
#define GTSAM_VERSION_STRING "4.3a0"

// Paths to example datasets distributed with GTSAM (unused by this extension;
// neutralized so the vendored header carries no build-machine path).
#define GTSAM_SOURCE_TREE_DATASET_DIR ""
#define GTSAM_INSTALLED_DATASET_DIR "/gtsam_examples/Data"

// Whether GTSAM is compiled to use quaternions for Rot3 (otherwise uses rotation matrices)
/* #undef GTSAM_USE_QUATERNIONS */

// Whether GTSAM is compiled to use Pose3::EXPMAP as the default coordinates mode for Pose3's retract and localCoordinates (otherwise, Pose3::FIRST_ORDER will be used)
#define GTSAM_POSE3_EXPMAP

// Whether GTSAM is compiled to use Rot3::EXPMAP as the default coordinates mode for Rot3's retract and localCoordinates (otherwise, Pose3::CAYLEY will be used)
#ifndef GTSAM_USE_QUATERNIONS
	#define GTSAM_ROT3_EXPMAP
#endif

// Whether to enable merging of equal leaf nodes in the Discrete Decision Tree.
#define GTSAM_DT_MERGING

// Whether to enable timing in hybrid factor graph machinery
#define GTSAM_HYBRID_TIMING 0

// Whether we are using TBB (if TBB was found and GTSAM_WITH_TBB is enabled in CMake)
/* #undef GTSAM_USE_TBB */

// Whether we are using a TBB version higher than 2020
/* #undef TBB_GREATER_EQUAL_2020 */

// Whether we are using system-Eigen or our own patched version
/* #undef GTSAM_USE_SYSTEM_EIGEN */

// Whether Eigen will use MKL (if MKL was found and GTSAM_WITH_EIGEN_MKL is enabled in CMake)
/* #undef GTSAM_USE_EIGEN_MKL */
/* #undef EIGEN_USE_MKL_ALL */

// Whether Eigen with MKL will use OpenMP (if OpenMP was found, Eigen uses MKL, and GTSAM_WITH_EIGEN_MKL_OPENMP is enabled in CMake)
/* #undef GTSAM_USE_EIGEN_MKL_OPENMP */

// Eigen library version (needed to avoid mixing versions, which often leads
// to segfaults)
#define GTSAM_EIGEN_VERSION_WORLD 3
#define GTSAM_EIGEN_VERSION_MAJOR 4
/* #undef GTSAM_EIGEN_VERSION_MINOR */

// The default allocator to use
/* #undef GTSAM_ALLOCATOR_BOOSTPOOL */
/* #undef GTSAM_ALLOCATOR_TBB */
#define GTSAM_ALLOCATOR_STL

// Option for not throwing the CheiralityException for points that are behind a camera
#define GTSAM_THROW_CHEIRALITY_EXCEPTION

// Make sure dependent projects that want it can see deprecated functions
#define GTSAM_ALLOW_DEPRECATED_SINCE_V43

// Support Metis-based nested dissection
#define GTSAM_SUPPORT_NESTED_DISSECTION

// Support Metis-based nested dissection
#define GTSAM_TANGENT_PREINTEGRATION

// Whether to use the system installed Metis instead of the provided one
/* #undef GTSAM_USE_SYSTEM_METIS */

// Toggle switch for BetweenFactor jacobian computation
/* #undef GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR */

/* #undef GTSAM_SLOW_BUT_CORRECT_EXPMAP */

// Boost flags
#define GTSAM_ENABLE_BOOST_SERIALIZATION 1
#define GTSAM_USE_BOOST_FEATURES 1
