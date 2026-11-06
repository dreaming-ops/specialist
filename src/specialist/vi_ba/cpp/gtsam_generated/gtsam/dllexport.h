/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file     dllexport.h
 * @brief    Symbols for exporting classes and methods from DLLs
 * @author   Richard Roberts
 * @date     Mar 9, 2013
 */

// Vendored copy of gtsam's cmake-generated dllexport.h (platform-independent).
// See specialist/vi_ba/cpp/gtsam_generated/gtsam/config.h for why.

#pragma once

// Whether GTSAM is compiled as static or DLL in windows.
// This will be used to decide whether include __declspec(dllimport) or not in headers
#define GTSAM_SHARED_LIB

#ifdef _WIN32
#  ifndef GTSAM_SHARED_LIB
#    define GTSAM_EXPORT
#    define GTSAM_EXTERN_EXPORT extern
#  else
#    ifdef GTSAM_EXPORTS
#      define GTSAM_EXPORT __declspec(dllexport)
#      define GTSAM_EXTERN_EXPORT __declspec(dllexport) extern
#    else
#      define GTSAM_EXPORT __declspec(dllimport)
#      define GTSAM_EXTERN_EXPORT __declspec(dllimport)
#    endif
#  endif
#else
#ifdef __APPLE__
#  define GTSAM_EXPORT __attribute__((visibility("default")))
#  define GTSAM_EXTERN_EXPORT extern
#else
#  define GTSAM_EXPORT
#  define GTSAM_EXTERN_EXPORT extern
#endif
#endif

// Keep implementation-detail symbols private to the library.
#if !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))
#  define GTSAM_LOCAL __attribute__((visibility("hidden")))
#else
#  define GTSAM_LOCAL
#endif

#undef GTSAM_SHARED_LIB
