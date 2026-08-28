//! Quaternion math — GLM-compatible `qua<scalar_type>`.
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

/// Quaternion type over scalar `scalar_type` (f32 or f64); fields are public so
/// `q.w`, `q.x`, ... can be read/written directly, and the default value
/// `.{}` is the identity quaternion. Complex numbers of rotation: two
/// quaternions equal up to a global sign represent the same rotation —
/// normalize and be consistent when comparing.
pub fn Quat(comptime scalar_type: type) type {
    return extern struct {
        const Self = @This();

        pub const value_type = scalar_type;
        pub const float_type = scalar.floatType(scalar_type);
        pub const len = 4;

        x: scalar_type = 0,
        y: scalar_type = 0,
        z: scalar_type = 0,
        w: scalar_type = 1,

        /// Identity quaternion (w = 1, no rotation). The starting point
        /// for `angleAxis`, multiplication chains and default camera
        /// orientation.
        pub inline fn identity() Self {
            return .{};
        }

        /// Build from four scalars, w (real part) FIRST (GLM
        /// `qua(w, x, y, z)`): `init(1, 0, 0, 0)` is the identity.
        pub inline fn init(w: scalar_type, x: scalar_type, y: scalar_type, z: scalar_type) Self {
            return .{ .w = w, .x = x, .y = y, .z = z };
        }

        /// Build from a scalar part plus a vector part (GLM
        /// `qua(s, v)`); the vector is the rotation axis of the
        /// corresponding rotation. `angleAxis` uses this internally.
        pub inline fn initScalarVec(w: scalar_type, vector_part: Vec(3, scalar_type)) Self {
            return .{ .w = w, .x = vector_part.v[0], .y = vector_part.v[1], .z = vector_part.v[2] };
        }

        /// Build from Euler angles in radians — p, y, r in order
        /// (GLM `qua(vec3 euler)`), using the standard XYZ convention:
        /// each component is halved, so build up with sinus products.
        /// Recover the original angles with `eulerAngles`.
        pub fn fromEuler(euler: Vec(3, scalar_type)) Self {
            const half = @as(scalar_type, 0.5);
            const c = Vec(3, scalar_type).init(.{
                scalar.cos(euler.v[0] * half),
                scalar.cos(euler.v[1] * half),
                scalar.cos(euler.v[2] * half),
            });
            const s = Vec(3, scalar_type).init(.{
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

        /// Right-handed look-at: the rotation quaternion that aligns the
        /// −z axis with `direction` (GLM `quatLookAtRH(direction, up)`).
        /// A producing const function — creates a new quaternion without
        /// touching any state. The degenerate `direction ∥ up` case is
        /// guarded by clamping the side-vector length instead of dividing
        /// by zero.
        pub fn lookAt(direction: Vec(3, scalar_type), up: Vec(3, scalar_type)) Self {
            return quatLookAt(direction, up);
        }

        /// Right-handed look-at: the rotation quaternion that aligns the
        /// −z axis with `direction` (GLM `quatLookAtRH(direction, up)`).
        /// Updates quaternion at pointer. The degenerate `direction ∥ up` case is
        /// guarded by clamping the side-vector length instead of dividing
        /// by zero.
        pub fn lookAtSelf(self: *Self, direction: Vec(3, scalar_type), up: Vec(3, scalar_type)) void {
            self.* = quatLookAt(direction, up);
        }

        // ---- component access ----

        /// Read component i (0..3, order x, y, z, w). For hot paths index
        /// the fields directly (`q.w`) — this switch is for generic code.
        pub inline fn get(self: Self, i: usize) scalar_type {
            return switch (i) {
                0 => self.x,
                1 => self.y,
                2 => self.z,
                3 => self.w,
                else => unreachable,
            };
        }

        /// Write component i (0..3, order x, y, z, w) of a mutable quaternion.
        pub inline fn set(self: *Self, i: usize, value: scalar_type) void {
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
        pub inline fn add(self: Self, right_hand_side: Self) Self {
            return .{ .w = self.w + right_hand_side.w, .x = self.x + right_hand_side.x, .y = self.y + right_hand_side.y, .z = self.z + right_hand_side.z };
        }

        /// In-place component-wise addition; `self` is overwritten with `add`.
        pub inline fn addSelf(self: *Self, right_hand_side: Self) void {
            self.* = self.add(right_hand_side);
        }

        /// Component-wise subtraction (GLM `operator-`); inverse of `add`.
        pub inline fn sub(self: Self, right_hand_side: Self) Self {
            return .{ .w = self.w - right_hand_side.w, .x = self.x - right_hand_side.x, .y = self.y - right_hand_side.y, .z = self.z - right_hand_side.z };
        }

        /// In-place component-wise subtraction; `self` is overwritten with `sub`.
        pub inline fn subSelf(self: *Self, right_hand_side: Self) void {
            self.* = self.sub(right_hand_side);
        }

        /// Component-wise negation (GLM unary `operator-`): represents the same
        /// rotation as `self` (globally); use inside `slerp` when the dot
        /// product is negative to pick the short arc.
        pub inline fn neg(self: Self) Self {
            return .{ .w = -self.w, .x = -self.x, .y = -self.y, .z = -self.z };
        }

        /// In-place negation; `self` is overwritten with `neg`.
        pub inline fn negSelf(self: *Self) void {
            self.* = self.neg();
        }

        /// Hamilton product (GLM `operator*`, quat * quat): the composition
        /// operator — `a.mul(b)` rotates by `b` first, then by `a`, like
        /// matrix products. This is the ONLY quaternion product that
        /// preserves the unit norm; normalize the result if the inputs
        /// drifted from unit length.
        pub inline fn mul(self: Self, right_hand_side: Self) Self {
            return .{
                .w = self.w * right_hand_side.w - self.x * right_hand_side.x - self.y * right_hand_side.y - self.z * right_hand_side.z,
                .x = self.w * right_hand_side.x + self.x * right_hand_side.w + self.y * right_hand_side.z - self.z * right_hand_side.y,
                .y = self.w * right_hand_side.y + self.y * right_hand_side.w + self.z * right_hand_side.x - self.x * right_hand_side.z,
                .z = self.w * right_hand_side.z + self.z * right_hand_side.w + self.x * right_hand_side.y - self.y * right_hand_side.x,
            };
        }

        /// In-place Hamilton product; `self` is overwritten with `mul`.
        pub inline fn mulSelf(self: *Self, right_hand_side: Self) void {
            self.* = self.mul(right_hand_side);
        }

        /// Scalar multiplication (GLM `operator*`, quat * scalar): scales all
        /// four components. With unit quaternions this is a smooth path
        /// toward zero (used by `mix`/`slerp`), not a rotation.
        pub inline fn mulScalar(self: Self, scalar_value: anytype) Self {
            const value = scalar.cast(scalar_type, scalar_value);
            return .{ .w = self.w * value, .x = self.x * value, .y = self.y * value, .z = self.z * value };
        }

        /// In-place scalar multiplication; `self` is overwritten with `mulScalar`.
        pub inline fn mulScalarSelf(self: *Self, scalar_value: anytype) void {
            self.* = self.mulScalar(scalar_value);
        }

        /// Scalar division (GLM `operator/`, quat / scalar): removes the scale
        /// a previous `mulScalar` introduced without a division per
        /// component — used by `normalize` and the slerp formulas.
        pub inline fn divScalar(self: Self, scalar_value: anytype) Self {
            const value = scalar.cast(scalar_type, scalar_value);
            return .{ .w = self.w / value, .x = self.x / value, .y = self.y / value, .z = self.z / value };
        }

        /// In-place scalar division; `self` is overwritten with `divScalar`.
        pub inline fn divScalarSelf(self: *Self, scalar_value: anytype) void {
            self.* = self.divScalar(scalar_value);
        }

        /// Rotate a 3D vector by this quaternion (GLM `operator*`, quat * vec3),
        /// via the optimized `v + 2·w·(qv×v) + 2·qv×(qv×v)` formatrix. `self`
        /// should be a UNIT quaternion — anything else scales the result.
        pub fn mulVec3(self: Self, vector: Vec(3, scalar_type)) Vec(3, scalar_type) {
            const QuatVector = Vec(3, scalar_type).init(.{ self.x, self.y, self.z });
            const uv = QuatVector.cross(vector);
            const uuv = QuatVector.cross(uv);
            return vector.add(uv.mul(self.w).add(uuv).mul(@as(scalar_type, 2)));
        }

        /// Rotate the positional part of a homogeneous 4-vector (GLM
        /// `operator*`, quat * vec4): `mulVec3` on (x, y, z), w is copied
        /// untouched — use to rotate points in clip/world space without
        /// touching the homogeneous coordinate.
        pub fn mulVec4(self: Self, vector: Vec(4, scalar_type)) Vec(4, scalar_type) {
            const r3 = self.mulVec3(Vec(3, scalar_type).init(.{ vector.v[0], vector.v[1], vector.v[2] }));
            return Vec(4, scalar_type).init(.{ r3.v[0], r3.v[1], r3.v[2], vector.v[3] });
        }

        // ---- geometric ----

        /// 4D dot product (GLM `dot(qua, qua)`). For unit quaternions the
        /// result is the cosine of half the rotation angle between them —
        /// the sign decides the short vs long interpolation arc.
        pub inline fn dot(self: Self, right_hand_side: Self) scalar_type {
            return self.w * right_hand_side.w + self.x * right_hand_side.x + self.y * right_hand_side.y + self.z * right_hand_side.z;
        }

        /// Length (norm) in 4D (GLM `length(qua)`): equals 1 for proper
        /// rotations. Values drifting from 1 signal accumulated drift from
        /// repeated `mul` — fix with `normalize`.
        pub inline fn length(self: Self) scalar_type {
            return scalar.sqrt(self.dot(self));
        }

        /// Renormalize to unit length (GLM `normalize(qua)`); the zero
        /// quaternion is replaced by the identity. Run after long
        /// multiplication chains to keep rotations exact — the drift is
        /// slow but accumulates.
        pub fn normalize(self: Self) Self {
            const norm = self.length();
            if (norm <= @as(scalar_type, 0)) return identity();
            return self.divScalar(norm);
        }

        /// In-place renormalization to unit length; `self` is overwritten with `normalize`.
        pub inline fn normalizeSelf(self: *Self) void {
            self.* = self.normalize();
        }

        /// Conjugate (GLM `conjugate(qua)`): flips the imaginary part. For unit
        /// quaternions this is the inverse rotation — `q.mul(q.conjugate())`
        /// is the identity, so use it to build difference quaternions
        /// `a.conjugate().mul(b)` ("rotate b as seen from a").
        pub inline fn conjugate(self: Self) Self {
            return .{ .w = self.w, .x = -self.x, .y = -self.y, .z = -self.z };
        }

        /// In-place conjugate; `self` is overwritten with `conjugate`.
        pub inline fn conjugateSelf(self: *Self) void {
            self.* = self.conjugate();
        }

        /// True inverse (GLM `inverse(qua)`): `conjugate / |q|²`. For unit
        /// quaternions `inverse` == `conjugate` (cheaper — prefer it);
        /// the full form matters only for non-unit quaternions driving
        /// similarity transforms.
        pub inline fn inverse(self: Self) Self {
            return self.conjugate().divScalar(self.dot(self));
        }

        /// In-place inverse; `self` is overwritten with `inverse`.
        pub inline fn inverseSelf(self: *Self) void {
            self.* = self.inverse();
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
    const scalar_type = scalar.rtType(@TypeOf(angleRad));
    const a = scalar.cast(scalar_type, angleRad);
    const s = scalar.sin(a * @as(scalar_type, 0.5));
    return Quat(scalar_type).initScalarVec(scalar.cos(a * @as(scalar_type, 0.5)), axisVec.mul(s));
}

/// Rotation angle in radians (GLM `angle(qua)`): `2·acos(w)` with special
/// handling near the identity — the result is in [0, 2π), and with
/// the short-arc normalization the angle is the minimal one.
pub fn angle(quaternion: anytype) @TypeOf(quaternion).value_type {
    const scalar_type = @TypeOf(quaternion).value_type;
    if (scalar.abs(quaternion.w) > scalar.cos(@as(scalar_type, 0.5))) {
        const a = scalar.asin(scalar.sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y + quaternion.z * quaternion.z)) * @as(scalar_type, 2);
        if (quaternion.w < @as(scalar_type, 0)) return scalar.cast(scalar_type, std.math.pi) * @as(scalar_type, 2) - a;
        return a;
    }
    return scalar.acos(quaternion.w) * @as(scalar_type, 2);
}

/// Rotation axis as a unit vector (GLM `axis(qua)`): the direction q
/// rotates about, derived from the normalized imaginary part
/// `(x, y, z) / sqrt(1 − w²)`; identity-like quaternions (|w| ≈ 1)
/// degenerate to +z, mirroring GLM. Together with `angle` this
/// reconstructs `angleAxis`.
pub fn axis(quaternion: anytype) Vec(3, @TypeOf(quaternion).value_type) {
    const scalar_type = @TypeOf(quaternion).value_type;
    const tmp1 = @as(scalar_type, 1) - quaternion.w * quaternion.w;
    if (tmp1 <= @as(scalar_type, 0)) return Vec(3, scalar_type).init(.{ 0, 0, 1 });
    const tmp2 = @as(scalar_type, 1) / scalar.sqrt(tmp1);
    return Vec(3, scalar_type).init(.{ quaternion.x * tmp2, quaternion.y * tmp2, quaternion.z * tmp2 });
}

/// Pitch angle in radians (GLM `pitch(qua)`): the X-axis component of
/// the rotation, extracted with GLM's exact atan2 formulas —
/// including its singularity handling (pure roll returns
/// `2·atan2(x, w)`). Combine with `yaw`/`roll` for debugging or
/// HUD displays; prefer the quaternion itself for logic.
pub fn pitch(quaternion: anytype) @TypeOf(quaternion).value_type {
    const scalar_type = @TypeOf(quaternion).value_type;
    const y = @as(scalar_type, 2) * (quaternion.y * quaternion.z + quaternion.w * quaternion.x);
    const x = quaternion.w * quaternion.w - quaternion.x * quaternion.x - quaternion.y * quaternion.y + quaternion.z * quaternion.z;
    if (y == @as(scalar_type, 0) and x == @as(scalar_type, 0)) return @as(scalar_type, 2) * scalar.atan2(quaternion.x, quaternion.w);
    return scalar.atan2(y, x);
}

/// Yaw angle in radians (GLM `yaw(qua)`): the Y-axis component, via
/// `asin` of a clamped expression — clamp keeps the domain valid
/// near the poles. Read it together with `pitch`/`roll` to
/// round-trip `fromEuler`.
pub fn yaw(quaternion: anytype) @TypeOf(quaternion).value_type {
    const scalar_type = @TypeOf(quaternion).value_type;
    const y = scalar.clamp(-@as(scalar_type, 2) * (quaternion.x * quaternion.z - quaternion.w * quaternion.y), -@as(scalar_type, 1), @as(scalar_type, 1));
    return scalar.asin(y);
}

/// Roll angle in radians (GLM `roll(qua)`): the Z-axis component, via
/// the same atan2 machinery as `pitch` (with its special case
/// returning 0 at the singularity).
pub fn roll(quaternion: anytype) @TypeOf(quaternion).value_type {
    const scalar_type = @TypeOf(quaternion).value_type;
    const y = @as(scalar_type, 2) * (quaternion.x * quaternion.y + quaternion.w * quaternion.z);
    const x = quaternion.w * quaternion.w + quaternion.x * quaternion.x - quaternion.y * quaternion.y - quaternion.z * quaternion.z;
    if (y == @as(scalar_type, 0) and x == @as(scalar_type, 0)) return @as(scalar_type, 0);
    return scalar.atan2(y, x);
}

/// Euler angles as a vector (GLM `eulerAngles(qua)`): `(pitch, yaw,
/// roll)` in radians, exactly GLM's order. This is a lossy
/// extraction near gimbal-lock poses — for animation blending keep
/// the quaternion and use `slerp`.
pub fn eulerAngles(quaternion: anytype) Vec(3, @TypeOf(quaternion).value_type) {
    const scalar_type = @TypeOf(quaternion).value_type;
    return Vec(3, scalar_type).init(.{ pitch(quaternion), yaw(quaternion), roll(quaternion) });
}

/// Convert to a 3x3 rotation matrix (GLM `mat3_cast(qua)`): the standard
/// Rodríguez expansion of the quaternion. Use when a shader or
/// physics engine wants a matrix; the matrix columns are the
/// rotated basis axes, and it is orthogonal for unit inputs.
pub fn mat3_cast(quaternion: anytype) Mat(3, 3, @TypeOf(quaternion).value_type) {
    const scalar_type = @TypeOf(quaternion).value_type;
    const qxx = quaternion.x * quaternion.x;
    const qyy = quaternion.y * quaternion.y;
    const qzz = quaternion.z * quaternion.z;
    const qxz = quaternion.x * quaternion.z;
    const qxy = quaternion.x * quaternion.y;
    const qyz = quaternion.y * quaternion.z;
    const qwx = quaternion.w * quaternion.x;
    const qwy = quaternion.w * quaternion.y;
    const qwz = quaternion.w * quaternion.z;

    const V3 = Vec(3, scalar_type);
    return Mat(3, 3, scalar_type).init(.{
        V3.init(.{ @as(scalar_type, 1) - @as(scalar_type, 2) * (qyy + qzz), @as(scalar_type, 2) * (qxy + qwz), @as(scalar_type, 2) * (qxz - qwy) }),
        V3.init(.{ @as(scalar_type, 2) * (qxy - qwz), @as(scalar_type, 1) - @as(scalar_type, 2) * (qxx + qzz), @as(scalar_type, 2) * (qyz + qwx) }),
        V3.init(.{ @as(scalar_type, 2) * (qxz + qwy), @as(scalar_type, 2) * (qyz - qwx), @as(scalar_type, 1) - @as(scalar_type, 2) * (qxx + qyy) }),
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
    const scalar_type = @TypeOf(matrix).value_type;
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

    const biggestVal = scalar.sqrt(fourBiggestSquaredMinus1 + @as(scalar_type, 1)) * @as(scalar_type, 0.5);
    const mult = @as(scalar_type, 0.25) / biggestVal;

    const Q = Quat(scalar_type);
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
    const scalar_type = @TypeOf(matrix).value_type;
    const V3 = Vec(3, scalar_type);
    const m3 = Mat(3, 3, scalar_type).init(.{
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
    const scalar_type = @TypeOf(quaternion).value_type;
    var tmp = axisVec;
    const norm = tmp.length();
    if (scalar.abs(norm - @as(scalar_type, 1)) > @as(scalar_type, 0.001)) {
        const oneOverLen = @as(scalar_type, 1) / norm;
        tmp = tmp.mul(oneOverLen);
    }
    return quaternion.mul(angleAxis(angleRad, tmp));
}

/// GLM `mix(qua, qua, a)`: component-wise `x·(1−a) + y·a` for near
/// parallel quaternions, otherwise a linear interpolation of the
/// sine components (a velocity-preserving variant of `slerp` that
/// does NOT normalize). Feed `factor` in [0, 1] and normalize the
/// result if you need a proper rotation.
pub fn mix(left_hand_side: anytype, right_hand_side: anytype, factor: @TypeOf(left_hand_side).value_type) @TypeOf(left_hand_side) {
    const scalar_type = @TypeOf(left_hand_side).value_type;
    const Q = @TypeOf(left_hand_side);
    const cosTheta = left_hand_side.dot(right_hand_side);
    if (cosTheta > @as(scalar_type, 1) - std.math.floatEps(scalar_type)) {
        return Q.init(scalar.mix(left_hand_side.w, right_hand_side.w, factor), scalar.mix(left_hand_side.x, right_hand_side.x, factor), scalar.mix(left_hand_side.y, right_hand_side.y, factor), scalar.mix(left_hand_side.z, right_hand_side.z, factor));
    }
    const angleRad = scalar.acos(cosTheta);
    return left_hand_side.mulScalar(scalar.sin((@as(scalar_type, 1) - factor) * angleRad)).add(right_hand_side.mulScalar(scalar.sin(factor * angleRad))).divScalar(scalar.sin(angleRad));
}

/// Linear interpolation (GLM `lerp`): plain `x·(1−a) + y·a` without
/// normalization or short-arc handling — the fastest blend, but
/// the magnitude shrinks mid-path and the path is not constant
/// speed. For animation use `slerp`; for tiny increments `lerp`
/// is fine.
pub fn lerp(left_hand_side: anytype, right_hand_side: anytype, factor: @TypeOf(left_hand_side).value_type) @TypeOf(left_hand_side) {
    const scalar_type = @TypeOf(left_hand_side).value_type;
    return left_hand_side.mulScalar(@as(scalar_type, 1) - factor).add(right_hand_side.mulScalar(factor));
}

/// Spherical linear interpolation (GLM `slerp(qua, qua, a)`): the
/// constant-angular-speed shortest path between the two rotations.
/// Flips `right_hand_side` when the dot product is negative (short arc) and
/// falls back to `mix` when parallel. THE standard animation and
/// camera blend; normalize `left_hand_side`/`right_hand_side` first for exactness.
pub fn slerp(left_hand_side: anytype, right_hand_side: anytype, factor: @TypeOf(left_hand_side).value_type) @TypeOf(left_hand_side) {
    const scalar_type = @TypeOf(left_hand_side).value_type;
    const Q = @TypeOf(left_hand_side);
    var z = right_hand_side;
    var cosTheta = left_hand_side.dot(right_hand_side);
    if (cosTheta < @as(scalar_type, 0)) {
        z = right_hand_side.neg();
        cosTheta = -cosTheta;
    }
    if (cosTheta > @as(scalar_type, 1) - std.math.floatEps(scalar_type)) {
        return Q.init(scalar.mix(left_hand_side.w, z.w, factor), scalar.mix(left_hand_side.x, z.x, factor), scalar.mix(left_hand_side.y, z.y, factor), scalar.mix(left_hand_side.z, z.z, factor));
    }
    const angleRad = scalar.acos(cosTheta);
    return left_hand_side.mulScalar(scalar.sin((@as(scalar_type, 1) - factor) * angleRad)).add(z.mulScalar(scalar.sin(factor * angleRad))).divScalar(scalar.sin(angleRad));
}

/// Slerp with an extra spin parameter `spin_count` (GLM `slerp(qua, qua, a, k)`,
/// per Graphics Gems III): the phase advances by `angle + spin_count·π`,
/// so `spin_count = 1` forces a full extra half-turn along the path. Use
/// for stylized barrel-roll interpolation that `slerp` cannot do.
pub fn slerpSpin(left_hand_side: anytype, right_hand_side: anytype, factor: @TypeOf(left_hand_side).value_type, spin_count: anytype) @TypeOf(left_hand_side) {
    const scalar_type = @TypeOf(left_hand_side).value_type;
    const Q = @TypeOf(left_hand_side);
    var z = right_hand_side;
    var cosTheta = left_hand_side.dot(right_hand_side);
    if (cosTheta < @as(scalar_type, 0)) {
        z = right_hand_side.neg();
        cosTheta = -cosTheta;
    }
    if (cosTheta > @as(scalar_type, 1) - std.math.floatEps(scalar_type)) {
        return Q.init(scalar.mix(left_hand_side.w, z.w, factor), scalar.mix(left_hand_side.x, z.x, factor), scalar.mix(left_hand_side.y, z.y, factor), scalar.mix(left_hand_side.z, z.z, factor));
    }
    const angleRad = scalar.acos(cosTheta);
    const phi = angleRad + @as(scalar_type, scalar.cast(scalar_type, spin_count)) * std.math.pi;
    return left_hand_side.mulScalar(scalar.sin(angleRad - factor * phi)).add(z.mulScalar(scalar.sin(factor * phi))).divScalar(scalar.sin(angleRad));
}

/// Quaternion exponential (GLM `exp(qua)`, ext/quaternion_exponential):
/// the analog of `exp` for rotations — `exp(0, 0, 0, θ)·axis` turns
/// the axis-angle form into a quaternion:
/// `exp({0, axis·θ}) = angleAxis(2θ, axis)`. Zero vector part
/// yields the identity (as e^0).
pub fn exp(quaternion: anytype) @TypeOf(quaternion) {
    const scalar_type = @TypeOf(quaternion).value_type;
    const Q = @TypeOf(quaternion);
    const u = Vec(3, scalar_type).init(.{ quaternion.x, quaternion.y, quaternion.z });
    const vec_angle = u.length();
    if (vec_angle < std.math.floatEps(scalar_type)) return Q.identity();
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
    const scalar_type = @TypeOf(quaternion).value_type;
    const Q = @TypeOf(quaternion);
    const u = Vec(3, scalar_type).init(.{ quaternion.x, quaternion.y, quaternion.z });
    const vec3_len = u.length();
    if (vec3_len < std.math.floatEps(scalar_type)) {
        if (quaternion.w > @as(scalar_type, 0)) return Q.initScalarVec(scalar.log(quaternion.w), Vec(3, scalar_type).zero());
        if (quaternion.w < @as(scalar_type, 0)) return Q.initScalarVec(scalar.log(-quaternion.w), Vec(3, scalar_type).init(.{ std.math.pi, 0, 0 }));
        return Q.init(std.math.inf(scalar_type), std.math.inf(scalar_type), std.math.inf(scalar_type), std.math.inf(scalar_type));
    }
    const t = scalar.atan2(vec3_len, quaternion.w) / vec3_len;
    const quat_len2 = vec3_len * vec3_len + quaternion.w * quaternion.w;
    return Q.initScalarVec(@as(scalar_type, 0.5) * scalar.log(quat_len2), Vec(3, scalar_type).init(.{ t * quaternion.x, t * quaternion.y, t * quaternion.z }));
}

/// Quaternion power (GLM `pow(qua, y)`, ext/quaternion_exponential):
/// `exp(y · log(q))`. For unit quaternions this scales the
/// rotation angle by `exponent` — `pow(q, 0.5)` is the square root
/// rotation. Near the identity (|w|/|q| > cos(0.5), i.e. rotation
/// angles below one radian) it switches to a numerically stable
/// asin-based branch, and `exponent ≈ 0` short-circuits to the identity.
pub fn pow(base: anytype, exponent: @TypeOf(base).value_type) @TypeOf(base) {
    const scalar_type = @TypeOf(base).value_type;
    const Q = @TypeOf(base);
    const eps = std.math.floatEps(scalar_type);
    if (exponent > -eps and exponent < eps) return Q.init(@as(scalar_type, 1), 0, 0, 0);
    const magnitude = scalar.sqrt(base.x * base.x + base.y * base.y + base.z * base.z + base.w * base.w);
    var ang: scalar_type = undefined;
    if (scalar.abs(base.w / magnitude) > @as(scalar_type, 0.877582561890372716130286068203503191)) {
        const vector_magnitude = base.x * base.x + base.y * base.y + base.z * base.z;
        if (vector_magnitude < std.math.floatMin(scalar_type)) {
            return Q.init(scalar.pow(base.w, exponent), 0, 0, 0);
        }
        ang = scalar.asin(scalar.sqrt(vector_magnitude) / magnitude);
    } else {
        ang = scalar.acos(base.w / magnitude);
    }
    const new_angle = ang * exponent;
    const div = scalar.sin(new_angle) / scalar.sin(ang);
    const mag = scalar.pow(magnitude, exponent - @as(scalar_type, 1));
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
pub fn equal(left_hand_side: anytype, right_hand_side: anytype) Vec(4, bool) {
    return Vec(4, bool).init(.{ left_hand_side.x == right_hand_side.x, left_hand_side.y == right_hand_side.y, left_hand_side.z == right_hand_side.z, left_hand_side.w == right_hand_side.w });
}

/// Tolerance equality (GLM `equal(qua, qua, eps)`): lane true iff
/// `|xᵢ − yᵢ| < eps`. Use for "same orientation within ε" checks
/// after `slerp`-driven animation settles.
pub fn equalEps(left_hand_side: anytype, right_hand_side: anytype, epsilon: @TypeOf(left_hand_side).value_type) Vec(4, bool) {
    const scalar_type = @TypeOf(left_hand_side).value_type;
    const v = Vec(4, scalar_type).init(.{ left_hand_side.x - right_hand_side.x, left_hand_side.y - right_hand_side.y, left_hand_side.z - right_hand_side.z, left_hand_side.w - right_hand_side.w });
    return Vec(4, bool).init(.{
        scalar.abs(v.v[0]) < epsilon,
        scalar.abs(v.v[1]) < epsilon,
        scalar.abs(v.v[2]) < epsilon,
        scalar.abs(v.v[3]) < epsilon,
    });
}

/// Exact component inequality as a 4-bool vector (GLM `notEqual(qua,
/// qua)`); the negation of `equal`.
pub fn notEqual(left_hand_side: anytype, right_hand_side: anytype) Vec(4, bool) {
    return Vec(4, bool).init(.{ left_hand_side.x != right_hand_side.x, left_hand_side.y != right_hand_side.y, left_hand_side.z != right_hand_side.z, left_hand_side.w != right_hand_side.w });
}

/// Tolerance inequality (GLM `notEqual(qua, qua, eps)`): lane true iff
/// `|xᵢ − yᵢ| ≥ eps`; the negation of `equalEps`.
pub fn notEqualEps(left_hand_side: anytype, right_hand_side: anytype, epsilon: @TypeOf(left_hand_side).value_type) Vec(4, bool) {
    const scalar_type = @TypeOf(left_hand_side).value_type;
    const v = Vec(4, scalar_type).init(.{ left_hand_side.x - right_hand_side.x, left_hand_side.y - right_hand_side.y, left_hand_side.z - right_hand_side.z, left_hand_side.w - right_hand_side.w });
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
    const scalar_type = @TypeOf(direction).value_type;
    const c2 = direction.neg();
    const right = up.cross(c2);
    const r0 = right.mul(scalar.inversesqrt(scalar.max(@as(scalar_type, 0.00001), right.dot(right))));
    const r1 = c2.cross(r0);
    return quat_cast(Mat(3, 3, scalar_type).init(.{ r0, r1, c2 }));
}

/// Left-handed look-at orientation (GLM `quatLookAtLH(direction, up)`,
/// gtc/quaternion): the twin of `quatLookAtRH` that aligns the +z
/// axis with `direction` — for engines where forward is +z.
pub fn quatLookAtLH(direction: anytype, up: anytype) Quat(@TypeOf(direction).value_type) {
    const scalar_type = @TypeOf(direction).value_type;
    const c2 = direction;
    const right = up.cross(c2);
    const r0 = right.mul(scalar.inversesqrt(scalar.max(@as(scalar_type, 0.00001), right.dot(right))));
    const r1 = c2.cross(r0);
    return quat_cast(Mat(3, 3, scalar_type).init(.{ r0, r1, c2 }));
}

/// Look-at orientation default (GLM `quatLookAt`): right-handed, for
/// parity with how this project's `lookAt` defaults; change to
/// `quatLookAtLH` if your convention is left-handed.
pub fn quatLookAt(direction: anytype, up: anytype) Quat(@TypeOf(direction).value_type) {
    return quatLookAtRH(direction, up);
}
