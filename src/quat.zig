//! Quaternion math — GLM-compatible `qua<T>`.
//!
//! A quaternion stores four components (x, y, z, w) where w is the REAL
//! (scalar) part and (x, y, z) the imaginary (vector) part — the same
//! layout as GLM. Unit quaternions represent 3D rotations without gimbal
//! lock and compose with the Hamilton product `mul`, so prefer them over
//! Euler angles for camera and rigid-body orientation, interpolating with
//! `slerp`.
//!
//! Conventions follow GLM: `w`-first construction, right-handed rotation
//! (`q.mulVec3(v)` rotates v about the axis part of q by `2·acos(w)`),
//! and the sign ambiguity is resolved exactly as GLM resolves it.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;

/// Quaternion type over scalar `T` (f32 or f64); fields are public so
/// `q.w`, `q.x`, ... can be read/written directly, and the default value
/// `.{}` is the identity quaternion. Complex numbers of rotation: two
/// quaternions equal up to a global sign represent the same rotation —
/// normalize and be consistent when comparing.
pub fn Quat(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const value_type = T;
        pub const float_type = scalar.floatType(T);
        pub const len = 4;

        x: T = 0,
        y: T = 0,
        z: T = 0,
        w: T = 1,

        /// Identity quaternion (w = 1, no rotation). The starting point
        /// for `angleAxis`, multiplication chains and default camera
        /// orientation.
        pub inline fn identity() Self {
            return .{};
        }

        /// Build from four scalars, w (real part) FIRST (GLM
        /// `qua(w, x, y, z)`): `init(1, 0, 0, 0)` is the identity.
        pub inline fn init(w: T, x: T, y: T, z: T) Self {
            return .{ .w = w, .x = x, .y = y, .z = z };
        }

        /// Build from a scalar part plus a vector part (GLM
        /// `qua(s, v)`); the vector is the rotation axis of the
        /// corresponding rotation. `angleAxis` uses this internally.
        pub inline fn initScalarVec(w: T, vector_part: Vec(3, T)) Self {
            return .{ .w = w, .x = vector_part.v[0], .y = vector_part.v[1], .z = vector_part.v[2] };
        }

        /// Build from Euler angles in radians — p, y, r in order
        /// (GLM `qua(vec3 euler)`), using the standard XYZ convention:
        /// each component is halved, so build up with sinus products.
        /// Recover the original angles with `eulerAngles`.
        pub fn fromEuler(euler: Vec(3, T)) Self {
            const half = @as(T, 0.5);
            const c = Vec(3, T).init(.{
                scalar.cos(euler.v[0] * half),
                scalar.cos(euler.v[1] * half),
                scalar.cos(euler.v[2] * half),
            });
            const s = Vec(3, T).init(.{
                scalar.sin(euler.v[0] * half),
                scalar.sin(euler.v[1] * half),
                scalar.sin(euler.v[2] * half),
            });
            return .{
                .w = c.v[0] * c.v[1] * c.v[2] + s.v[0] * s.v[1] * s.v[2],
                .x = s.v[0] * c.v[1] * c.v[2] - c.v[0] * s.v[1] * s.v[2],
                .y = c.v[0] * s.v[1] * c.v[2] + s.v[0] * c.v[1] * s.v[2],
                .z = c.v[0] * c.v[1] * s.v[2] - s.v[0] * s.v[1] * c.v[2],
            };
        }

        // ---- component access ----

        /// Read component i (0..3, order x, y, z, w). For hot paths index
        /// the fields directly (`q.w`) — this switch is for generic code.
        pub inline fn get(self: Self, i: usize) T {
            return switch (i) {
                0 => self.x,
                1 => self.y,
                2 => self.z,
                3 => self.w,
                else => unreachable,
            };
        }

        /// Write component i (0..3, order x, y, z, w) of a mutable quaternion.
        pub inline fn set(self: *Self, i: usize, value: T) void {
            switch (i) {
                0 => self.x = value,
                1 => self.y = value,
                2 => self.z = value,
                3 => self.w = value,
                else => unreachable,
            }
        }

        // ---- arithmetic ----

        /// Component-wise addition (GLM `operator+`). Rare in rotation
        /// work — quaternions add meaningfully only in interpolation
        /// formulas (see `mix`), which are computed via `mulScalar`.
        pub inline fn add(self: Self, rhs: Self) Self {
            return .{ .w = self.w + rhs.w, .x = self.x + rhs.x, .y = self.y + rhs.y, .z = self.z + rhs.z };
        }

        /// Component-wise subtraction (GLM `operator-`); inverse of `add`.
        pub inline fn sub(self: Self, rhs: Self) Self {
            return .{ .w = self.w - rhs.w, .x = self.x - rhs.x, .y = self.y - rhs.y, .z = self.z - rhs.z };
        }

        /// Component-wise negation (GLM unary `operator-`): represents the same
        /// rotation as `self` (globally); use inside `slerp` when the dot
        /// product is negative to pick the short arc.
        pub inline fn neg(self: Self) Self {
            return .{ .w = -self.w, .x = -self.x, .y = -self.y, .z = -self.z };
        }

        /// Hamilton product (GLM `operator*`, quat * quat): the composition
        /// operator — `a.mul(b)` rotates by `b` first, then by `a`, like
        /// matrix products. This is the ONLY quaternion product that
        /// preserves the unit norm; normalize the result if the inputs
        /// drifted from unit length.
        pub inline fn mul(self: Self, rhs: Self) Self {
            return .{
                .w = self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                .x = self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                .y = self.w * rhs.y + self.y * rhs.w + self.z * rhs.x - self.x * rhs.z,
                .z = self.w * rhs.z + self.z * rhs.w + self.x * rhs.y - self.y * rhs.x,
            };
        }

        /// Scalar multiplication (GLM `operator*`, quat * scalar): scales all
        /// four components. With unit quaternions this is a smooth path
        /// toward zero (used by `mix`/`slerp`), not a rotation.
        pub inline fn mulScalar(self: Self, scalar_value: anytype) Self {
            const value = scalar.cast(T, scalar_value);
            return .{ .w = self.w * value, .x = self.x * value, .y = self.y * value, .z = self.z * value };
        }

        /// Scalar division (GLM `operator/`, quat / scalar): removes the scale
        /// a previous `mulScalar` introduced without a division per
        /// component — used by `normalize` and the slerp formulas.
        pub inline fn divScalar(self: Self, scalar_value: anytype) Self {
            const value = scalar.cast(T, scalar_value);
            return .{ .w = self.w / value, .x = self.x / value, .y = self.y / value, .z = self.z / value };
        }

        /// Rotate a 3D vector by this quaternion (GLM `operator*`, quat * vec3),
        /// via the optimized `v + 2·w·(qv×v) + 2·qv×(qv×v)` formatrix. `self`
        /// should be a UNIT quaternion — anything else scales the result.
        pub fn mulVec3(self: Self, vector: Vec(3, T)) Vec(3, T) {
            const QuatVector = Vec(3, T).init(.{ self.x, self.y, self.z });
            const uv = QuatVector.cross(vector);
            const uuv = QuatVector.cross(uv);
            return vector.add(uv.mul(self.w).add(uuv).mul(@as(T, 2)));
        }

        /// Rotate the positional part of a homogeneous 4-vector (GLM
        /// `operator*`, quat * vec4): `mulVec3` on (x, y, z), w is copied
        /// untouched — use to rotate points in clip/world space without
        /// touching the homogeneous coordinate.
        pub fn mulVec4(self: Self, vector: Vec(4, T)) Vec(4, T) {
            const r3 = self.mulVec3(Vec(3, T).init(.{ vector.v[0], vector.v[1], vector.v[2] }));
            return Vec(4, T).init(.{ r3.v[0], r3.v[1], r3.v[2], vector.v[3] });
        }

        // ---- geometric ----

        /// 4D dot product (GLM `dot(qua, qua)`). For unit quaternions the
        /// result is the cosine of half the rotation angle between them —
        /// the sign decides the short vs long interpolation arc.
        pub inline fn dot(self: Self, rhs: Self) T {
            return self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z;
        }

        /// Length (norm) in 4D (GLM `length(qua)`): equals 1 for proper
        /// rotations. Values drifting from 1 signal accumulated drift from
        /// repeated `mul` — fix with `normalize`.
        pub inline fn length(self: Self) T {
            return scalar.sqrt(self.dot(self));
        }

        /// Renormalize to unit length (GLM `normalize(qua)`); the zero
        /// quaternion is replaced by the identity. Run after long
        /// multiplication chains to keep rotations exact — the drift is
        /// slow but accumulates.
        pub fn normalize(self: Self) Self {
            const norm = self.length();
            if (norm <= @as(T, 0)) return identity();
            return self.divScalar(norm);
        }

        /// Conjugate (GLM `conjugate(qua)`): flips the imaginary part. For unit
        /// quaternions this is the inverse rotation — `q.mul(q.conjugate())`
        /// is the identity, so use it to build difference quaternions
        /// `a.conjugate().mul(b)` ("rotate b as seen from a").
        pub inline fn conjugate(self: Self) Self {
            return .{ .w = self.w, .x = -self.x, .y = -self.y, .z = -self.z };
        }

        /// True inverse (GLM `inverse(qua)`): `conjugate / |q|²`. For unit
        /// quaternions `inverse` == `conjugate` (cheaper — prefer it);
        /// the full form matters only for non-unit quaternions driving
        /// similarity transforms.
        pub inline fn inverse(self: Self) Self {
            return self.conjugate().divScalar(self.dot(self));
        }

        /// `{any}` formatter: prints `{w,x,y,z}` with `{d}` floats, e.g.
        /// `{0.70710678,0,0,0.70710678}`. Enables `std.debug.print(
        /// "{any}", .{q})` in logs and tests.
        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("{{{d},{d},{d},{d}}}", .{ self.w, self.x, self.y, self.z });
        }
    };
}

// ---- free functions (GLM namespace-level) ----

/// Build a rotation quaternion: `angleRad` around `axisVec` (GLM
/// `angleAxis(angle, axis)`, angle in radians). Equivalent to rotating
/// about a normalized axis by the given angle — `q.mulVec3(any_v)` then
/// spins `any_v` that way. The axis is NOT normalized internally; pass a
/// unit vector unless you intend a scaled rotation.
pub fn angleAxis(angleRad: anytype, axisVec: Vec(3, scalar.rtType(@TypeOf(angleRad)))) Quat(scalar.rtType(@TypeOf(angleRad))) {
    const T = scalar.rtType(@TypeOf(angleRad));
    const a = scalar.cast(T, angleRad);
    const s = scalar.sin(a * @as(T, 0.5));
    return Quat(T).initScalarVec(scalar.cos(a * @as(T, 0.5)), axisVec.mul(s));
}

/// Rotation angle in radians (GLM `angle(qua)`): `2·acos(w)` with special
        /// handling near the identity — the result is in [0, 2π), and with
        /// the short-arc normalization the angle is the minimal one.
        pub fn angle(quaternion: anytype) @TypeOf(quaternion).value_type {
    const T = @TypeOf(quaternion).value_type;
    if (scalar.abs(quaternion.w) > scalar.cos(@as(T, 0.5))) {
        const a = scalar.asin(scalar.sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y + quaternion.z * quaternion.z)) * @as(T, 2);
        if (quaternion.w < @as(T, 0)) return scalar.cast(T, std.math.pi) * @as(T, 2) - a;
        return a;
    }
    return scalar.acos(quaternion.w) * @as(T, 2);
}

/// Rotation axis as a unit vector (GLM `axis(qua)`): the direction q
        /// rotates about, derived from the normalized imaginary part
        /// `(x, y, z) / sqrt(1 − w²)`; identity-like quaternions (|w| ≈ 1)
        /// degenerate to +z, mirroring GLM. Together with `angle` this
        /// reconstructs `angleAxis`.
        pub fn axis(quaternion: anytype) Vec(3, @TypeOf(quaternion).value_type) {
    const T = @TypeOf(quaternion).value_type;
    const tmp1 = @as(T, 1) - quaternion.w * quaternion.w;
    if (tmp1 <= @as(T, 0)) return Vec(3, T).init(.{ 0, 0, 1 });
    const tmp2 = @as(T, 1) / scalar.sqrt(tmp1);
    return Vec(3, T).init(.{ quaternion.x * tmp2, quaternion.y * tmp2, quaternion.z * tmp2 });
}

/// Pitch angle in radians (GLM `pitch(qua)`): the X-axis component of
        /// the rotation, extracted with GLM's exact atan2 formulas —
        /// including its singularity handling (pure roll returns
        /// `2·atan2(x, w)`). Combine with `yaw`/`roll` for debugging or
        /// HUD displays; prefer the quaternion itself for logic.
        pub fn pitch(quaternion: anytype) @TypeOf(quaternion).value_type {
    const T = @TypeOf(quaternion).value_type;
    const y = @as(T, 2) * (quaternion.y * quaternion.z + quaternion.w * quaternion.x);
    const x = quaternion.w * quaternion.w - quaternion.x * quaternion.x - quaternion.y * quaternion.y + quaternion.z * quaternion.z;
    if (y == @as(T, 0) and x == @as(T, 0)) return @as(T, 2) * scalar.atan2(quaternion.x, quaternion.w);
    return scalar.atan2(y, x);
}

/// Yaw angle in radians (GLM `yaw(qua)`): the Y-axis component, via
        /// `asin` of a clamped expression — clamp keeps the domain valid
        /// near the poles. Read it together with `pitch`/`roll` to
        /// round-trip `fromEuler`.
        pub fn yaw(quaternion: anytype) @TypeOf(quaternion).value_type {
    const T = @TypeOf(quaternion).value_type;
    const y = scalar.clamp(-@as(T, 2) * (quaternion.x * quaternion.z - quaternion.w * quaternion.y), -@as(T, 1), @as(T, 1));
    return scalar.asin(y);
}

/// Roll angle in radians (GLM `roll(qua)`): the Z-axis component, via
        /// the same atan2 machinery as `pitch` (with its special case
        /// returning 0 at the singularity).
        pub fn roll(quaternion: anytype) @TypeOf(quaternion).value_type {
    const T = @TypeOf(quaternion).value_type;
    const y = @as(T, 2) * (quaternion.x * quaternion.y + quaternion.w * quaternion.z);
    const x = quaternion.w * quaternion.w + quaternion.x * quaternion.x - quaternion.y * quaternion.y - quaternion.z * quaternion.z;
    if (y == @as(T, 0) and x == @as(T, 0)) return @as(T, 0);
    return scalar.atan2(y, x);
}

/// Euler angles as a vector (GLM `eulerAngles(qua)`): `(pitch, yaw,
        /// roll)` in radians, exactly GLM's order. This is a lossy
        /// extraction near gimbal-lock poses — for animation blending keep
        /// the quaternion and use `slerp`.
        pub fn eulerAngles(quaternion: anytype) Vec(3, @TypeOf(quaternion).value_type) {
    const T = @TypeOf(quaternion).value_type;
    return Vec(3, T).init(.{ pitch(quaternion), yaw(quaternion), roll(quaternion) });
}

/// Convert to a 3x3 rotation matrix (GLM `mat3_cast(qua)`): the standard
        /// Rodríguez expansion of the quaternion. Use when a shader or
        /// physics engine wants a matrix; the matrix columns are the
        /// rotated basis axes, and it is orthogonal for unit inputs.
        pub fn mat3_cast(quaternion: anytype) Mat(3, 3, @TypeOf(quaternion).value_type) {
    const T = @TypeOf(quaternion).value_type;
    const qxx = quaternion.x * quaternion.x;
    const qyy = quaternion.y * quaternion.y;
    const qzz = quaternion.z * quaternion.z;
    const qxz = quaternion.x * quaternion.z;
    const qxy = quaternion.x * quaternion.y;
    const qyz = quaternion.y * quaternion.z;
    const qwx = quaternion.w * quaternion.x;
    const qwy = quaternion.w * quaternion.y;
    const qwz = quaternion.w * quaternion.z;

    const V3 = Vec(3, T);
    return Mat(3, 3, T).init(.{
        V3.init(.{ @as(T, 1) - @as(T, 2) * (qyy + qzz), @as(T, 2) * (qxy + qwz), @as(T, 2) * (qxz - qwy) }),
        V3.init(.{ @as(T, 2) * (qxy - qwz), @as(T, 1) - @as(T, 2) * (qxx + qzz), @as(T, 2) * (qyz + qwx) }),
        V3.init(.{ @as(T, 2) * (qxz + qwy), @as(T, 2) * (qyz - qwx), @as(T, 1) - @as(T, 2) * (qxx + qyy) }),
    });
}

/// Convert to a 4x4 rotation matrix (GLM `mat4_cast(qua)`): `mat3_cast`
        /// embedded in the upper-left 3x3 of a homogeneous identity — the
        /// usual way to feed a quaternion rotation into a transform
        /// pipeline (after which `translate`/`scale` compose normally).
        pub fn mat4_cast(quaternion: anytype) Mat(4, 4, @TypeOf(quaternion).value_type) {
    return mat3_cast(quaternion).toMat4();
}

/// Convert a 3x3 rotation matrix to a quaternion (GLM `quat_cast(mat3)`):
        /// the Shepperd method — picks the largest diagonal term for
        /// numerical stability (so it works even with noisy matrices),
        /// and is the inverse of `mat3_cast` for proper rotations.
        /// Mirrored/invalid matrices yield a quaternion representing the
        /// nearest rotation.
        pub fn quat_cast(matrix: anytype) Quat(@TypeOf(matrix).value_type) {
    const T = @TypeOf(matrix).value_type;
    const fourXSquaredMinus1 = matrix.data[0].v[0] - matrix.data[1].v[1] - matrix.data[2].v[2];
    const fourYSquaredMinus1 = matrix.data[1].v[1] - matrix.data[0].v[0] - matrix.data[2].v[2];
    const fourZSquaredMinus1 = matrix.data[2].v[2] - matrix.data[0].v[0] - matrix.data[1].v[1];
    const fourWSquaredMinus1 = matrix.data[0].v[0] + matrix.data[1].v[1] + matrix.data[2].v[2];

    var biggestIndex: usize = 0;
    var fourBiggestSquaredMinus1 = fourWSquaredMinus1;
    if (fourXSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourXSquaredMinus1;
        biggestIndex = 1;
    }
    if (fourYSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourYSquaredMinus1;
        biggestIndex = 2;
    }
    if (fourZSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourZSquaredMinus1;
        biggestIndex = 3;
    }

    const biggestVal = scalar.sqrt(fourBiggestSquaredMinus1 + @as(T, 1)) * @as(T, 0.5);
    const mult = @as(T, 0.25) / biggestVal;

    const Q = Quat(T);
    return switch (biggestIndex) {
        0 => Q.init(
            biggestVal,
            (matrix.data[1].v[2] - matrix.data[2].v[1]) * mult,
            (matrix.data[2].v[0] - matrix.data[0].v[2]) * mult,
            (matrix.data[0].v[1] - matrix.data[1].v[0]) * mult,
        ),
        1 => Q.init(
            (matrix.data[1].v[2] - matrix.data[2].v[1]) * mult,
            biggestVal,
            (matrix.data[0].v[1] + matrix.data[1].v[0]) * mult,
            (matrix.data[2].v[0] + matrix.data[0].v[2]) * mult,
        ),
        2 => Q.init(
            (matrix.data[2].v[0] - matrix.data[0].v[2]) * mult,
            (matrix.data[0].v[1] + matrix.data[1].v[0]) * mult,
            biggestVal,
            (matrix.data[1].v[2] + matrix.data[2].v[1]) * mult,
        ),
        3 => Q.init(
            (matrix.data[0].v[1] - matrix.data[1].v[0]) * mult,
            (matrix.data[2].v[0] + matrix.data[0].v[2]) * mult,
            (matrix.data[1].v[2] + matrix.data[2].v[1]) * mult,
            biggestVal,
        ),
        else => unreachable,
    };
}

/// Convert a 4x4 matrix's rotation to a quaternion (GLM `quat_cast(mat4)`):
        /// extracts the upper-left 3x3 and delegates to `quat_cast` —
        /// translation, scale and the fourth column are ignored, so scale
        /// the matrix's linear part first (or use `affineInverse`-style
        /// decomposition) for non-rigid transforms.
        pub fn quat_cast4(matrix: anytype) Quat(@TypeOf(matrix).value_type) {
    const T = @TypeOf(matrix).value_type;
    const V3 = Vec(3, T);
    const m3 = Mat(3, 3, T).init(.{
        V3.init(.{ matrix.data[0].v[0], matrix.data[0].v[1], matrix.data[0].v[2] }),
        V3.init(.{ matrix.data[1].v[0], matrix.data[1].v[1], matrix.data[1].v[2] }),
        V3.init(.{ matrix.data[2].v[0], matrix.data[2].v[1], matrix.data[2].v[2] }),
    });
    return quat_cast(m3);
}

/// Append a rotation: `q * angleAxis(angle, normalize(axis))` (GLM
        /// `rotate(qua, angle, axis)`). The axis is normalized on the fly
        /// (only if it drifted by more than 0.001 from unit length, per
        /// GLM), so passing a near-unit axis is fine.
        pub fn rotate(quaternion: anytype, angleRad: anytype, axisVec: Vec(3, @TypeOf(quaternion).value_type)) @TypeOf(quaternion) {
    const T = @TypeOf(quaternion).value_type;
    var tmp = axisVec;
    const norm = tmp.length();
    if (scalar.abs(norm - @as(T, 1)) > @as(T, 0.001)) {
        const oneOverLen = @as(T, 1) / norm;
        tmp = tmp.mul(oneOverLen);
    }
    return quaternion.mul(angleAxis(angleRad, tmp));
}

/// GLM `mix(qua, qua, a)`: component-wise `x·(1−a) + y·a` for near
        /// parallel quaternions, otherwise a linear interpolation of the
        /// sine components (a velocity-preserving variant of `slerp` that
        /// does NOT normalize). Feed `a` in [0, 1] and normalize the
        /// result if you need a proper rotation.
        pub fn mix(lhs: anytype, rhs: anytype, factor: @TypeOf(lhs).value_type) @TypeOf(lhs) {
    const T = @TypeOf(lhs).value_type;
    const Q = @TypeOf(lhs);
    const cosTheta = lhs.dot(rhs);
    if (cosTheta > @as(T, 1) - std.math.floatEps(T)) {
        return Q.init(scalar.mix(lhs.w, rhs.w, factor), scalar.mix(lhs.x, rhs.x, factor), scalar.mix(lhs.y, rhs.y, factor), scalar.mix(lhs.z, rhs.z, factor));
    }
    const angleRad = scalar.acos(cosTheta);
    return lhs.mulScalar(scalar.sin((@as(T, 1) - factor) * angleRad)).add(rhs.mulScalar(scalar.sin(factor * angleRad))).divScalar(scalar.sin(angleRad));
}

/// Linear interpolation (GLM `lerp`): plain `x·(1−a) + y·a` without
        /// normalization or short-arc handling — the fastest blend, but
        /// the magnitude shrinks mid-path and the path is not constant
        /// speed. For animation use `slerp`; for tiny increments `lerp`
        /// is fine.
        pub fn lerp(lhs: anytype, rhs: anytype, factor: @TypeOf(lhs).value_type) @TypeOf(lhs) {
    const T = @TypeOf(lhs).value_type;
    return lhs.mulScalar(@as(T, 1) - factor).add(rhs.mulScalar(factor));
}

/// Spherical linear interpolation (GLM `slerp(qua, qua, a)`): the
        /// constant-angular-speed shortest path between the two rotations.
        /// Flips `y` when the dot product is negative (short arc) and
        /// falls back to `mix` when parallel. THE standard animation and
        /// camera blend; normalize `x`/`y` first for exactness.
        pub fn slerp(lhs: anytype, rhs: anytype, factor: @TypeOf(lhs).value_type) @TypeOf(lhs) {
    const T = @TypeOf(lhs).value_type;
    const Q = @TypeOf(lhs);
    var z = rhs;
    var cosTheta = lhs.dot(rhs);
    if (cosTheta < @as(T, 0)) {
        z = rhs.neg();
        cosTheta = -cosTheta;
    }
    if (cosTheta > @as(T, 1) - std.math.floatEps(T)) {
        return Q.init(scalar.mix(lhs.w, z.w, factor), scalar.mix(lhs.x, z.x, factor), scalar.mix(lhs.y, z.y, factor), scalar.mix(lhs.z, z.z, factor));
    }
    const angleRad = scalar.acos(cosTheta);
    return lhs.mulScalar(scalar.sin((@as(T, 1) - factor) * angleRad)).add(z.mulScalar(scalar.sin(factor * angleRad))).divScalar(scalar.sin(angleRad));
}

/// Slerp with an extra spin parameter `k` (GLM `slerp(qua, qua, a, k)`,
        /// per Graphics Gems III): the phase advances by `angle + k·π`,
        /// so `k = 1` forces a full extra half-turn along the path. Use
        /// for stylized barrel-roll interpolation that `slerp` cannot do.
        pub fn slerpSpin(lhs: anytype, rhs: anytype, factor: @TypeOf(lhs).value_type, spin_count: anytype) @TypeOf(lhs) {
    const T = @TypeOf(lhs).value_type;
    const Q = @TypeOf(lhs);
    var z = rhs;
    var cosTheta = lhs.dot(rhs);
    if (cosTheta < @as(T, 0)) {
        z = rhs.neg();
        cosTheta = -cosTheta;
    }
    if (cosTheta > @as(T, 1) - std.math.floatEps(T)) {
        return Q.init(scalar.mix(lhs.w, z.w, factor), scalar.mix(lhs.x, z.x, factor), scalar.mix(lhs.y, z.y, factor), scalar.mix(lhs.z, z.z, factor));
    }
    const angleRad = scalar.acos(cosTheta);
    const phi = angleRad + @as(T, scalar.cast(T, spin_count)) * std.math.pi;
    return lhs.mulScalar(scalar.sin(angleRad - factor * phi)).add(z.mulScalar(scalar.sin(factor * phi))).divScalar(scalar.sin(angleRad));
}

/// Quaternion exponential (GLM `exp(qua)`, ext/quaternion_exponential):
        /// the analog of `exp` for rotations — `exp(0, 0, 0, θ)·axis` turns
        /// the axis-angle form into a quaternion:
        /// `exp({0, axis·θ}) = angleAxis(2θ, axis)`. Zero vector part
        /// yields the identity (as e^0).
        pub fn exp(quaternion: anytype) @TypeOf(quaternion) {
    const T = @TypeOf(quaternion).value_type;
    const Q = @TypeOf(quaternion);
    const u = Vec(3, T).init(.{ quaternion.x, quaternion.y, quaternion.z });
    const vec_angle = u.length();
    if (vec_angle < std.math.floatEps(T)) return Q.identity();
    const v = u.div(vec_angle);
    return Q.initScalarVec(scalar.cos(vec_angle), v.mul(scalar.sin(vec_angle)));
}

/// Quaternion logarithm (GLM `log(qua)`, ext/quaternion_exponential):
        /// the inverse of `exp` — maps a rotation back to its
        /// axis-scaled-angle form `(0, axis·θ)`, logging the magnitude on
        /// top. The w < 0 branch maps to the +x axis with angle π (the
        /// antipodal quaternion) and the degenerate zero quaternion maps
        /// to infinities, matching GLM exactly.
        pub fn log(quaternion: anytype) @TypeOf(quaternion) {
    const T = @TypeOf(quaternion).value_type;
    const Q = @TypeOf(quaternion);
    const u = Vec(3, T).init(.{ quaternion.x, quaternion.y, quaternion.z });
    const vec3_len = u.length();
    if (vec3_len < std.math.floatEps(T)) {
        if (quaternion.w > @as(T, 0)) return Q.initScalarVec(scalar.log(quaternion.w), Vec(3, T).zero());
        if (quaternion.w < @as(T, 0)) return Q.initScalarVec(scalar.log(-quaternion.w), Vec(3, T).init(.{ std.math.pi, 0, 0 }));
        return Q.init(std.math.inf(T), std.math.inf(T), std.math.inf(T), std.math.inf(T));
    }
    const t = scalar.atan2(vec3_len, quaternion.w) / vec3_len;
    const quat_len2 = vec3_len * vec3_len + quaternion.w * quaternion.w;
    return Q.initScalarVec(@as(T, 0.5) * scalar.log(quat_len2), Vec(3, T).init(.{ t * quaternion.x, t * quaternion.y, t * quaternion.z }));
}

/// Quaternion power (GLM `pow(qua, y)`, ext/quaternion_exponential):
        /// `exp(y · log(q))`. For unit quaternions this scales the
        /// rotation angle by `y` — `pow(q, 0.5)` is the square root
        /// rotation. Near the identity (|w|/|q| > cos(0.5), i.e. rotation
        /// angles below one radian) it switches to a numerically stable
        /// asin-based branch, and `y ≈ 0` short-circuits to the identity.
        pub fn pow(base: anytype, exponent: @TypeOf(base).value_type) @TypeOf(base) {
    const T = @TypeOf(base).value_type;
    const Q = @TypeOf(base);
    const eps = std.math.floatEps(T);
    if (exponent > -eps and exponent < eps) return Q.init(@as(T, 1), 0, 0, 0);
    const magnitude = scalar.sqrt(base.x * base.x + base.y * base.y + base.z * base.z + base.w * base.w);
    var ang: T = undefined;
    if (scalar.abs(base.w / magnitude) > @as(T, 0.877582561890372716130286068203503191)) {
        const vector_magnitude = base.x * base.x + base.y * base.y + base.z * base.z;
        if (vector_magnitude < std.math.floatMin(T)) {
            return Q.init(scalar.pow(base.w, exponent), 0, 0, 0);
        }
        ang = scalar.asin(scalar.sqrt(vector_magnitude) / magnitude);
    } else {
        ang = scalar.acos(base.w / magnitude);
    }
    const new_angle = ang * exponent;
    const div = scalar.sin(new_angle) / scalar.sin(ang);
    const mag = scalar.pow(magnitude, exponent - @as(T, 1));
    return Q.init(scalar.cos(new_angle) * magnitude * mag, base.x * div * mag, base.y * div * mag, base.z * div * mag);
}

/// Square root of the rotation (GLM `sqrt(qua)`): `pow(q, 0.5)` — the
        /// half-angle quaternion, e.g. to split a single keyframe rotation
        /// into two identical steps for an animation curve.
        pub fn sqrt(quaternion: anytype) @TypeOf(quaternion) {
    return pow(quaternion, @as(@TypeOf(quaternion).value_type, 0.5));
}

/// Exact component equality as a 4-bool vector (GLM `equal(qua, qua)`,
        /// ext/quaternion_relational): lanes are true where components
        /// are bitwise equal. Two quaternions negated represent the same
        /// rotation — compare via `dot(x, y).abs() > 1 - eps` instead if
        /// you mean "same orientation".
        pub fn equal(lhs: anytype, rhs: anytype) Vec(4, bool) {
    return Vec(4, bool).init(.{ lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w });
}

/// Tolerance equality (GLM `equal(qua, qua, eps)`): lane true iff
        /// `|xᵢ − yᵢ| < eps`. Use for "same orientation within ε" checks
        /// after `slerp`-driven animation settles.
        pub fn equalEps(lhs: anytype, rhs: anytype, epsilon: @TypeOf(lhs).value_type) Vec(4, bool) {
    const T = @TypeOf(lhs).value_type;
    const v = Vec(4, T).init(.{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w });
    return Vec(4, bool).init(.{
        scalar.abs(v.v[0]) < epsilon,
        scalar.abs(v.v[1]) < epsilon,
        scalar.abs(v.v[2]) < epsilon,
        scalar.abs(v.v[3]) < epsilon,
    });
}

/// Exact component inequality as a 4-bool vector (GLM `notEqual(qua,
        /// qua)`); the negation of `equal`.
        pub fn notEqual(lhs: anytype, rhs: anytype) Vec(4, bool) {
    return Vec(4, bool).init(.{ lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w });
}

/// Tolerance inequality (GLM `notEqual(qua, qua, eps)`): lane true iff
        /// `|xᵢ − yᵢ| ≥ eps`; the negation of `equalEps`.
        pub fn notEqualEps(lhs: anytype, rhs: anytype, epsilon: @TypeOf(lhs).value_type) Vec(4, bool) {
    const T = @TypeOf(lhs).value_type;
    const v = Vec(4, T).init(.{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w });
    return Vec(4, bool).init(.{
        scalar.abs(v.v[0]) >= epsilon,
        scalar.abs(v.v[1]) >= epsilon,
        scalar.abs(v.v[2]) >= epsilon,
        scalar.abs(v.v[3]) >= epsilon,
    });
}

/// Per-component NaN test (GLM `isnan(qua)`, ext/quaternion_common) as a
        /// bool vector — combine with `.any()` to detect invalid
        /// orientations (e.g. from dividing by a zero-length axis) during
        /// simulation debug.
        pub fn isnan(quaternion: anytype) Vec(4, bool) {
    return Vec(4, bool).init(.{ std.math.isNan(quaternion.x), std.math.isNan(quaternion.y), std.math.isNan(quaternion.z), std.math.isNan(quaternion.w) });
}

/// Per-component infinity test (GLM `isinf(qua)`) as a bool vector —
        /// flag quaternions corrupted by `log` of the zero quaternion or
        /// by exploding interpolations.
        pub fn isinf(quaternion: anytype) Vec(4, bool) {
    return Vec(4, bool).init(.{ std.math.isInf(quaternion.x), std.math.isInf(quaternion.y), std.math.isInf(quaternion.z), std.math.isInf(quaternion.w) });
}

/// Right-handed look-at orientation (GLM `quatLookAtRH(direction, up)`,
        /// gtc/quaternion): the quaternion that rotates the −z axis to
        /// point along `direction` — the rotation part of `lookAt`.
        /// Degenerate `direction ∥ up` cases are guarded by clamping the
        /// side-vector length instead of normalizing by zero.
        pub fn quatLookAtRH(direction: anytype, up: anytype) Quat(@TypeOf(direction).value_type) {
    const T = @TypeOf(direction).value_type;
    const c2 = direction.neg();
    const right = up.cross(c2);
    const r0 = right.mul(scalar.inversesqrt(scalar.max(@as(T, 0.00001), right.dot(right))));
    const r1 = c2.cross(r0);
    return quat_cast(Mat(3, 3, T).init(.{ r0, r1, c2 }));
}

/// Left-handed look-at orientation (GLM `quatLookAtLH(direction, up)`,
        /// gtc/quaternion): the twin of `quatLookAtRH` that aligns the +z
        /// axis with `direction` — for engines where forward is +z.
        pub fn quatLookAtLH(direction: anytype, up: anytype) Quat(@TypeOf(direction).value_type) {
    const T = @TypeOf(direction).value_type;
    const c2 = direction;
    const right = up.cross(c2);
    const r0 = right.mul(scalar.inversesqrt(scalar.max(@as(T, 0.00001), right.dot(right))));
    const r1 = c2.cross(r0);
    return quat_cast(Mat(3, 3, T).init(.{ r0, r1, c2 }));
}

/// Look-at orientation default (GLM `quatLookAt`): right-handed, for
        /// parity with how this project's `lookAt` defaults; change to
        /// `quatLookAtLH` if your convention is left-handed.
        pub fn quatLookAt(direction: anytype, up: anytype) Quat(@TypeOf(direction).value_type) {
    return quatLookAtRH(direction, up);
}
