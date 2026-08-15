const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;

/// GLM `qua<T>` — quaternion. Storage order x, y, z, w (like GLM, whose
/// constructors take the scalar part first).
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

        pub inline fn identity() Self {
            return .{};
        }

        /// GLM `qua(w, x, y, z)` — scalar-first.
        pub inline fn init(w: T, x: T, y: T, z: T) Self {
            return .{ .w = w, .x = x, .y = y, .z = z };
        }

        /// GLM `qua(s, v)` — scalar + vec3 part.
        pub inline fn initScalarVec(w: T, v: Vec(3, T)) Self {
            return .{ .w = w, .x = v.v[0], .y = v.v[1], .z = v.v[2] };
        }

        /// GLM `qua(vec3 euler)` — from Euler angles (radians).
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

        pub inline fn get(self: Self, i: usize) T {
            return switch (i) {
                0 => self.x,
                1 => self.y,
                2 => self.z,
                3 => self.w,
                else => unreachable,
            };
        }

        pub inline fn set(self: *Self, i: usize, v: T) void {
            switch (i) {
                0 => self.x = v,
                1 => self.y = v,
                2 => self.z = v,
                3 => self.w = v,
                else => unreachable,
            }
        }

        // ---- arithmetic ----

        /// GLM `operator+`.
        pub inline fn add(self: Self, q: Self) Self {
            return .{ .w = self.w + q.w, .x = self.x + q.x, .y = self.y + q.y, .z = self.z + q.z };
        }

        /// GLM `operator-`.
        pub inline fn sub(self: Self, q: Self) Self {
            return .{ .w = self.w - q.w, .x = self.x - q.x, .y = self.y - q.y, .z = self.z - q.z };
        }

        /// GLM unary `operator-`.
        pub inline fn neg(self: Self) Self {
            return .{ .w = -self.w, .x = -self.x, .y = -self.y, .z = -self.z };
        }

        /// GLM `operator*` (quat * quat) — Hamilton product.
        pub inline fn mul(self: Self, q: Self) Self {
            return .{
                .w = self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
                .x = self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
                .y = self.w * q.y + self.y * q.w + self.z * q.x - self.x * q.z,
                .z = self.w * q.z + self.z * q.w + self.x * q.y - self.y * q.x,
            };
        }

        /// GLM `operator*` (quat * scalar).
        pub inline fn mulScalar(self: Self, s: anytype) Self {
            const v = scalar.cast(T, s);
            return .{ .w = self.w * v, .x = self.x * v, .y = self.y * v, .z = self.z * v };
        }

        /// GLM `operator/` (quat / scalar).
        pub inline fn divScalar(self: Self, s: anytype) Self {
            const v = scalar.cast(T, s);
            return .{ .w = self.w / v, .x = self.x / v, .y = self.y / v, .z = self.z / v };
        }

        /// GLM `operator*` (quat * vec3) — rotate the vector.
        pub fn mulVec3(self: Self, v: Vec(3, T)) Vec(3, T) {
            const QuatVector = Vec(3, T).init(.{ self.x, self.y, self.z });
            const uv = QuatVector.cross(v);
            const uuv = QuatVector.cross(uv);
            return v.add(uv.mul(self.w).add(uuv).mul(@as(T, 2)));
        }

        /// GLM `operator*` (quat * vec4) — vec4(quat * vec3(v), v.w).
        pub fn mulVec4(self: Self, v: Vec(4, T)) Vec(4, T) {
            const r3 = self.mulVec3(Vec(3, T).init(.{ v.v[0], v.v[1], v.v[2] }));
            return Vec(4, T).init(.{ r3.v[0], r3.v[1], r3.v[2], v.v[3] });
        }

        // ---- geometric ----

        /// GLM `dot(qua, qua)`.
        pub inline fn dot(self: Self, q: Self) T {
            return self.w * q.w + self.x * q.x + self.y * q.y + self.z * q.z;
        }

        /// GLM `length(qua)`.
        pub inline fn length(self: Self) T {
            return scalar.sqrt(self.dot(self));
        }

        /// GLM `normalize(qua)`.
        pub fn normalize(self: Self) Self {
            const norm = self.length();
            if (norm <= @as(T, 0)) return identity();
            return self.divScalar(norm);
        }

        /// GLM `conjugate(qua)`.
        pub inline fn conjugate(self: Self) Self {
            return .{ .w = self.w, .x = -self.x, .y = -self.y, .z = -self.z };
        }

        /// GLM `inverse(qua)`.
        pub inline fn inverse(self: Self) Self {
            return self.conjugate().divScalar(self.dot(self));
        }

        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("{{{d},{d},{d},{d}}}", .{ self.w, self.x, self.y, self.z });
        }
    };
}

// ---- free functions (GLM namespace-level) ----

/// GLM `angleAxis(angle, axis)` — angle in radians.
pub fn angleAxis(angleRad: anytype, axisVec: Vec(3, scalar.rtType(@TypeOf(angleRad)))) Quat(scalar.rtType(@TypeOf(angleRad))) {
    const T = scalar.rtType(@TypeOf(angleRad));
    const a = scalar.cast(T, angleRad);
    const s = scalar.sin(a * @as(T, 0.5));
    return Quat(T).initScalarVec(scalar.cos(a * @as(T, 0.5)), axisVec.mul(s));
}

/// GLM `angle(qua)` — rotation angle in radians.
pub fn angle(q: anytype) @TypeOf(q).value_type {
    const T = @TypeOf(q).value_type;
    if (scalar.abs(q.w) > scalar.cos(@as(T, 0.5))) {
        const a = scalar.asin(scalar.sqrt(q.x * q.x + q.y * q.y + q.z * q.z)) * @as(T, 2);
        if (q.w < @as(T, 0)) return scalar.cast(T, std.math.pi) * @as(T, 2) - a;
        return a;
    }
    return scalar.acos(q.w) * @as(T, 2);
}

/// GLM `axis(qua)` — rotation axis.
pub fn axis(q: anytype) Vec(3, @TypeOf(q).value_type) {
    const T = @TypeOf(q).value_type;
    const tmp1 = @as(T, 1) - q.w * q.w;
    if (tmp1 <= @as(T, 0)) return Vec(3, T).init(.{ 0, 0, 1 });
    const tmp2 = @as(T, 1) / scalar.sqrt(tmp1);
    return Vec(3, T).init(.{ q.x * tmp2, q.y * tmp2, q.z * tmp2 });
}

/// GLM `pitch(qua)`.
pub fn pitch(q: anytype) @TypeOf(q).value_type {
    const T = @TypeOf(q).value_type;
    const y = @as(T, 2) * (q.y * q.z + q.w * q.x);
    const x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;
    if (y == @as(T, 0) and x == @as(T, 0)) return @as(T, 2) * scalar.atan2(q.x, q.w);
    return scalar.atan2(y, x);
}

/// GLM `yaw(qua)`.
pub fn yaw(q: anytype) @TypeOf(q).value_type {
    const T = @TypeOf(q).value_type;
    const y = scalar.clamp(-@as(T, 2) * (q.x * q.z - q.w * q.y), -@as(T, 1), @as(T, 1));
    return scalar.asin(y);
}

/// GLM `roll(qua)`.
pub fn roll(q: anytype) @TypeOf(q).value_type {
    const T = @TypeOf(q).value_type;
    const y = @as(T, 2) * (q.x * q.y + q.w * q.z);
    const x = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;
    if (y == @as(T, 0) and x == @as(T, 0)) return @as(T, 0);
    return scalar.atan2(y, x);
}

/// GLM `eulerAngles(qua)` — vec3(pitch, yaw, roll).
pub fn eulerAngles(q: anytype) Vec(3, @TypeOf(q).value_type) {
    const T = @TypeOf(q).value_type;
    return Vec(3, T).init(.{ pitch(q), yaw(q), roll(q) });
}

/// GLM `mat3_cast(qua)`.
pub fn mat3_cast(q: anytype) Mat(3, 3, @TypeOf(q).value_type) {
    const T = @TypeOf(q).value_type;
    const qxx = q.x * q.x;
    const qyy = q.y * q.y;
    const qzz = q.z * q.z;
    const qxz = q.x * q.z;
    const qxy = q.x * q.y;
    const qyz = q.y * q.z;
    const qwx = q.w * q.x;
    const qwy = q.w * q.y;
    const qwz = q.w * q.z;

    const V3 = Vec(3, T);
    return Mat(3, 3, T).init(.{
        V3.init(.{ @as(T, 1) - @as(T, 2) * (qyy + qzz), @as(T, 2) * (qxy + qwz), @as(T, 2) * (qxz - qwy) }),
        V3.init(.{ @as(T, 2) * (qxy - qwz), @as(T, 1) - @as(T, 2) * (qxx + qzz), @as(T, 2) * (qyz + qwx) }),
        V3.init(.{ @as(T, 2) * (qxz + qwy), @as(T, 2) * (qyz - qwx), @as(T, 1) - @as(T, 2) * (qxx + qyy) }),
    });
}

/// GLM `mat4_cast(qua)`.
pub fn mat4_cast(q: anytype) Mat(4, 4, @TypeOf(q).value_type) {
    return mat3_cast(q).toMat4();
}

/// GLM `quat_cast(mat3)`.
pub fn quat_cast(m: anytype) Quat(@TypeOf(m).value_type) {
    const T = @TypeOf(m).value_type;
    const fourXSquaredMinus1 = m.data[0].v[0] - m.data[1].v[1] - m.data[2].v[2];
    const fourYSquaredMinus1 = m.data[1].v[1] - m.data[0].v[0] - m.data[2].v[2];
    const fourZSquaredMinus1 = m.data[2].v[2] - m.data[0].v[0] - m.data[1].v[1];
    const fourWSquaredMinus1 = m.data[0].v[0] + m.data[1].v[1] + m.data[2].v[2];

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
            (m.data[1].v[2] - m.data[2].v[1]) * mult,
            (m.data[2].v[0] - m.data[0].v[2]) * mult,
            (m.data[0].v[1] - m.data[1].v[0]) * mult,
        ),
        1 => Q.init(
            (m.data[1].v[2] - m.data[2].v[1]) * mult,
            biggestVal,
            (m.data[0].v[1] + m.data[1].v[0]) * mult,
            (m.data[2].v[0] + m.data[0].v[2]) * mult,
        ),
        2 => Q.init(
            (m.data[2].v[0] - m.data[0].v[2]) * mult,
            (m.data[0].v[1] + m.data[1].v[0]) * mult,
            biggestVal,
            (m.data[1].v[2] + m.data[2].v[1]) * mult,
        ),
        3 => Q.init(
            (m.data[0].v[1] - m.data[1].v[0]) * mult,
            (m.data[2].v[0] + m.data[0].v[2]) * mult,
            (m.data[1].v[2] + m.data[2].v[1]) * mult,
            biggestVal,
        ),
        else => unreachable,
    };
}

/// GLM `quat_cast(mat4)`.
pub fn quat_cast4(m: anytype) Quat(@TypeOf(m).value_type) {
    const T = @TypeOf(m).value_type;
    const V3 = Vec(3, T);
    const m3 = Mat(3, 3, T).init(.{
        V3.init(.{ m.data[0].v[0], m.data[0].v[1], m.data[0].v[2] }),
        V3.init(.{ m.data[1].v[0], m.data[1].v[1], m.data[1].v[2] }),
        V3.init(.{ m.data[2].v[0], m.data[2].v[1], m.data[2].v[2] }),
    });
    return quat_cast(m3);
}

/// GLM `rotate(qua, angle, axis)` — `q * angleAxis(angle, normalize(axis))`.
pub fn rotate(q: anytype, angleRad: anytype, axisVec: Vec(3, @TypeOf(q).value_type)) @TypeOf(q) {
    const T = @TypeOf(q).value_type;
    var tmp = axisVec;
    const norm = tmp.length();
    if (scalar.abs(norm - @as(T, 1)) > @as(T, 0.001)) {
        const oneOverLen = @as(T, 1) / norm;
        tmp = tmp.mul(oneOverLen);
    }
    return q.mul(angleAxis(angleRad, tmp));
}

/// GLM `mix(qua, qua, a)`.
pub fn mix(x: anytype, y: anytype, a: @TypeOf(x).value_type) @TypeOf(x) {
    const T = @TypeOf(x).value_type;
    const Q = @TypeOf(x);
    const cosTheta = x.dot(y);
    if (cosTheta > @as(T, 1) - std.math.floatEps(T)) {
        return Q.init(scalar.mix(x.w, y.w, a), scalar.mix(x.x, y.x, a), scalar.mix(x.y, y.y, a), scalar.mix(x.z, y.z, a));
    }
    const angleRad = scalar.acos(cosTheta);
    return x.mulScalar(scalar.sin((@as(T, 1) - a) * angleRad)).add(y.mulScalar(scalar.sin(a * angleRad))).divScalar(scalar.sin(angleRad));
}

/// GLM `lerp(qua, qua, a)`.
pub fn lerp(x: anytype, y: anytype, a: @TypeOf(x).value_type) @TypeOf(x) {
    const T = @TypeOf(x).value_type;
    return x.mulScalar(@as(T, 1) - a).add(y.mulScalar(a));
}

/// GLM `slerp(qua, qua, a)`.
pub fn slerp(x: anytype, y: anytype, a: @TypeOf(x).value_type) @TypeOf(x) {
    const T = @TypeOf(x).value_type;
    const Q = @TypeOf(x);
    var z = y;
    var cosTheta = x.dot(y);
    if (cosTheta < @as(T, 0)) {
        z = y.neg();
        cosTheta = -cosTheta;
    }
    if (cosTheta > @as(T, 1) - std.math.floatEps(T)) {
        return Q.init(scalar.mix(x.w, z.w, a), scalar.mix(x.x, z.x, a), scalar.mix(x.y, z.y, a), scalar.mix(x.z, z.z, a));
    }
    const angleRad = scalar.acos(cosTheta);
    return x.mulScalar(scalar.sin((@as(T, 1) - a) * angleRad)).add(z.mulScalar(scalar.sin(a * angleRad))).divScalar(scalar.sin(angleRad));
}
