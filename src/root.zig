//! ZigSimpleEngine math library — a GLM 1.1.0-compatible math port.
//!
//! Modules:
//! - `scalar` — scalar math & helpers (float/int dispatch, trig,
//!   exponential, bit ops, ULP comparisons, rounding utilities),
//! - `vec` — vector math: `Vec(component_count, scalar_type)` (vec2/vec3/vec4-class vectors) with
//!   per-component and geometric operations,
//! - `mat` — matrix math: `Mat(num_columns, num_rows, scalar_type)` (column-major, GLM layout),
//!   transforms and the perspective/ortho/frustum clip-space family,
//! - `quat` — quaternions: `Quat(T)` with rotation composition,
//!   interpolation and conversions,
//! - `constants` — mathematical constants.
//!
//! Style matches GLM: methods read receiver-first (GLM free functions),
//! storage is column-major and quaternions are w-first — the
//! same conventions as the GLM headers this library is verified against.

const std = @import("std");

/// Scalar math module (see `scalar.zig` for the full function list).
pub const scalar = @import("scalar.zig");
/// Vector module: `Vec(L, T)` plus the `isVec` type test.
pub const vec = @import("vec.zig");
/// Matrix module: `Mat(C, R, T)` plus clip-space/transform helpers.
pub const mat = @import("mat.zig");
/// Quaternion module: `Quat(T)` plus angle/Euler/matrix conversions.
pub const quat = @import("quat.zig");

/// Vector type alias: `Vec(3, f32)` is vec3, `Vec(4, i32)` is ivec4...
/// Instantiate via `vec.Vec(L, T)` or this shortcut.
pub const Vec = vec.Vec;
/// Matrix type alias: `Mat(4, 4, f32)` is mat4; shape is (columns, rows).
pub const Mat = mat.Mat;
/// Quaternion type alias: `Quat(f32)` is the single-precision quaternion.
pub const Quat = quat.Quat;
