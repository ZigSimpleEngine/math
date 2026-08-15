//! Validation tests against reference values generated from original GLM 1.1.0
//! (see C:\Users\DanP1e\AppData\Local\Temp\opencode\glm_ref).

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const quat = @import("quat.zig");
const constants = @import("constants.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;
const Quat = quat.Quat;
const vec2 = Vec(2, f32);
const vec3 = Vec(3, f32);
const vec4 = Vec(4, f32);
const bvec3 = Vec(3, bool);
const ivec3 = Vec(3, i32);
const ivec4 = Vec(4, i32);
const mat2 = Mat(2, 2, f32);
const mat3 = Mat(3, 3, f32);
const mat4 = Mat(4, 4, f32);

fn expectMat4Approx(actual: anytype, expected: [16]f32, tol: f32) !void {
    inline for (0..4) |c| inline for (0..4) |r| {
        try std.testing.expectApproxEqAbs(expected[c * 4 + r], @as(f32, @floatCast(actual.data[c].v[r])), tol);
    };
}

fn expectVecApprox(actual: anytype, expected: anytype, tol: f32) !void {
    const A = @TypeOf(actual);
    inline for (0..A.len) |i| {
        try std.testing.expectApproxEqAbs(
            @as(f32, @floatCast(@as(@TypeOf(expected).value_type, expected.v[i]))),
            @as(f32, @floatCast(actual.v[i])),
            tol,
        );
    }
}

fn vec3_(x: f32, y: f32, z: f32) vec3 {
    return vec3.init(.{ x, y, z });
}

// ---- common scalars ----

test "scalar common" {
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), scalar.abs(-3.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), scalar.sign(-2.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), scalar.floor(2.7), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2), scalar.ceil(-2.3), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), scalar.round(2.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), scalar.roundEven(3.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), scalar.roundEven(2.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), scalar.roundEven(4.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2), scalar.trunc(-2.7), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.700000048), scalar.fract(-2.3), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.30000019), scalar.mod(5.3, 2.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), scalar.min(1.5, -2.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), scalar.max(1.5, -2.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scalar.clamp(2.5, 0.0, 1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), scalar.mix(1.0, 3.0, 0.25), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), scalar.step(1.0, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), scalar.smoothstep(0.0, 1.0, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), scalar.fma(2.0, 3.0, 1.0), 1e-6);
}

test "scalar modf frexp ldexp" {
    const m = scalar.modf(5.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), m.fract, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5), m.integral, 1e-6);
    const f = scalar.frexp(@as(f32, 12.5));
    try std.testing.expectApproxEqAbs(@as(f32, 0.78125), f.significand, 1e-6);
    try std.testing.expectEqual(@as(i32, 4), f.exponent);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), scalar.ldexp(@as(f32, 0.78125), 4), 1e-6);
}

test "constants" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0000001192), constants.epsilon(f32), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14159274), constants.pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.877582561), constants.cos_one_over_two(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.28318548), constants.two_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.77245385), constants.root_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.57079637), constants.half_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.71238898), constants.three_over_two_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.785398185), constants.quarter_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.318309873), constants.one_over_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.159154937), constants.one_over_two_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.636619747), constants.two_over_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.27323949), constants.four_over_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.12837923), constants.two_over_root_pi(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.707106769), constants.one_over_root_two(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.41421354), constants.root_two(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.73205078), constants.root_three(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.23606798), constants.root_five(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.71828175), constants.e(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.577215672), constants.euler(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.693147182), constants.ln_two(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.30258512), constants.ln_ten(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.366512924), constants.ln_ln_two(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.618034), constants.golden_ratio(f32), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14159265358979323846), constants.pi(f64), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 2.71828182845904523536), constants.e(f64), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.577215664901532860606), constants.euler(f64), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 1.61803398874989484820), constants.golden_ratio(f64), 1e-15);
}

test "scalar trig exp" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.57079637), scalar.radians(90.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 90), scalar.degrees(1.57079632679), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8), scalar.pow(@as(f32, 2.0), @as(f32, 3.0)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11.3137083), scalar.exp2(3.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.58496261), scalar.log2(12.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), scalar.inversesqrt(16.0), 1e-6);
}

// ---- vec3 geometric ----

test "vec geometric" {
    const a = vec3_(1.0, 2.0, 3.0);
    const b = vec3_(-1.0, 0.5, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 6), a.dot(b), 1e-6);
    try expectVecApprox(a.cross(b), vec3_(2.5, -5, 2.5), 1e-6);
    try expectVecApprox(a.normalize(), vec3_(0.267261237, 0.534522474, 0.801783681), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.7416575), a.length(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.69258237), a.distance(b), 1e-6);
    try expectVecApprox(a.reflect(b.normalize()), vec3_(3.28571415, 0.857142925, -1.5714283), 1e-6);
    try expectVecApprox(a.refract(b.normalize(), 0.66), vec3_(2.23675251, 0.531623781, -1.17350507), 1e-6);
    try expectVecApprox(a.faceforward(b, vec3_(0.0, 1.0, 0.0)), vec3_(-1, -2, -3), 1e-6);
}

// ---- vec3 common ----

test "vec common" {
    const b = vec3_(-1.0, 0.5, 2.0);
    try expectVecApprox(b.abs(), vec3_(1, 0.5, 2), 1e-6);
    try expectVecApprox(vec3_(2.5, -1.5, 0.5).clamp(0.0, 1.0), vec3_(1, 0, 0.5), 1e-6);
    try expectVecApprox(vec3.fill(1).mix(vec3.fill(3), vec3_(0.25, 0.5, 0.75)), vec3_(1.5, 2, 2.5), 1e-6);
    try expectVecApprox(vec3.fill(1).mix(vec3.fill(3), 0.5), vec3_(2, 2, 2), 1e-6);
    try expectVecApprox(vec3_(0.25, 0.5, 0.75).smoothstep(0.0, 1.0), vec3_(0.15625, 0.5, 0.84375), 1e-6);
    try expectVecApprox(vec3_(0.25, 0.5, 0.75).step(0.5), vec3_(0, 1, 1), 1e-6);
    try expectVecApprox(vec3_(-1.5, 2.25, 3.75).fract(), vec3_(0.5, 0.25, 0.75), 1e-6);
    try expectVecApprox(vec3_(5.3, 2.0, -1.0).mod(vec3_(2.0, 1.5, 3.0)), vec3_(1.30000019, 0.5, 2), 1e-6);
    try expectVecApprox(vec3.fill(2).pow(vec3_(2.0, 3.0, 4.0)), vec3_(4, 8, 16), 1e-6);
    try expectVecApprox(vec3_(0.5, 1.0, 2.0).exp(), vec3_(1.64872122, 2.71828175, 7.38905621), 1e-6);
    try expectVecApprox(vec3_(0.5, 1.0, 2.0).log(), vec3_(-0.693147182, 0, 0.693147182), 1e-6);
    try expectVecApprox(vec3_(0.0, 1.0, 2.0).sin(), vec3_(0, 0.841470957, 0.909297407), 1e-6);
    try expectVecApprox(vec3_(0.0, 1.0, 2.0).cos(), vec3_(1, 0.540302277, -0.416146845), 1e-6);
    try expectVecApprox(vec3_(1.0, 0.5, 0.25).atan(), vec3_(0.785398185, 0.463647604, 0.244978666), 1e-6);
    try expectVecApprox(vec3_(1.0, 0.5, 0.25).atan2(vec3.fill(1)), vec3_(0.785398185, 0.463647604, 0.244978666), 1e-6);
    try expectVecApprox(vec3_(4.0, 9.0, 16.0).sqrt(), vec3_(2, 3, 4), 1e-6);
    try expectVecApprox(vec3_(1.0, 5.0, -2.0).min(vec3_(2.0, 3.0, 0.0)), vec3_(1, 3, -2), 1e-6);
    try expectVecApprox(vec3_(1.0, 5.0, -2.0).max(vec3_(2.0, 3.0, 0.0)), vec3_(2, 5, 0), 1e-6);
    try expectVecApprox(vec3_(0.5, 1.5, 2.5).roundEven(), vec3_(0, 2, 2), 1e-6);
    try expectVecApprox(vec3_(0.0, 10.0, 20.0).mix(vec3_(2.0, 14.0, 30.0), 0.5), vec3_(1, 12, 25), 1e-6);
    try expectVecApprox(vec3_(-2.0, 0.0, 3.0).sign(), vec3_(-1, 0, 1), 1e-6);
}

test "vec relational int" {
    try std.testing.expect(bvec3.init(.{ true, false, false }).any());
    try std.testing.expect(bvec3.init(.{ true, true, true }).all());
    try expectVecApprox(vec3.init(ivec3.init(.{ -1, 2, -3 }).abs()), vec3_(1, 2, 3), 1e-6);
    const lt = vec3_(1.0, 2.0, 3.0).lessThan(vec3_(2.0, 2.0, 2.0));
    try std.testing.expectEqual(true, lt.x());
    try std.testing.expectEqual(false, lt.y());
    try std.testing.expectEqual(false, lt.z());
    const eqv = vec3_(1.0, 2.0, 3.0).equal(vec3_(1.0, 5.0, 3.0));
    try std.testing.expectEqual(true, eqv.x());
    try std.testing.expectEqual(false, eqv.y());
    try std.testing.expectEqual(true, eqv.z());
    const gte = vec3_(1.0, 2.0, 3.0).greaterThanEqual(vec3_(1.0, 2.0, 5.0));
    try std.testing.expectEqual(true, gte.x());
    try std.testing.expectEqual(true, gte.y());
    try std.testing.expectEqual(false, gte.z());
}

// ---- matrices ----

fn m4seq() mat4 {
    // GLM: mat4(1..16) fills column 0 with (1,2,3,4), column 1 with (5,6,7,8), ...
    return mat4.init(.{
        vec4.init(.{ 1, 2, 3, 4 }),
        vec4.init(.{ 5, 6, 7, 8 }),
        vec4.init(.{ 9, 10, 11, 12 }),
        vec4.init(.{ 13, 14, 15, 16 }),
    });
}

test "mat2" {
    const m2 = mat2.init(.{ vec2.init(.{ 1, 2 }), vec2.init(.{ 3, 4 }) });
    try std.testing.expectApproxEqAbs(@as(f32, -2), m2.determinant(), 1e-6);
    const inv = m2.inverse();
    try std.testing.expectApproxEqAbs(@as(f32, -2), inv.data[0].v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), inv.data[0].v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), inv.data[1].v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), inv.data[1].v[1], 1e-6);
}

test "mat3" {
    const m3 = mat3.init(.{
        vec3.init(.{ 1, 2, 3 }),
        vec3.init(.{ 0, 1, 4 }),
        vec3.init(.{ 5, 6, 0 }),
    });
    try std.testing.expectApproxEqAbs(@as(f32, 1), m3.determinant(), 1e-6);
    const inv = m3.inverse();
    const exp = [_]f32{ -24, 18, 5, 20, -15, -4, -5, 4, 1 };
    inline for (0..3) |c| inline for (0..3) |r| {
        try std.testing.expectApproxEqAbs(exp[c * 3 + r], inv.data[c].v[r], 1e-6);
    };
    const t = m3.transpose();
    const texp = [_]f32{ 1, 0, 5, 2, 1, 6, 3, 4, 0 };
    inline for (0..3) |c| inline for (0..3) |r| {
        try std.testing.expectApproxEqAbs(texp[c * 3 + r], t.data[c].v[r], 1e-6);
    };
}

test "mat4" {
    const m4 = m4seq();
    try std.testing.expectApproxEqAbs(@as(f32, 0), m4.determinant(), 1e-6);
    const inv = m4.inverse();
    inline for (0..4) |c| inline for (0..4) |r| {
        try std.testing.expect(std.math.isNan(inv.data[c].v[r]));
    };
    const t = m4.transpose();
    const texp = [_]f32{ 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 };
    inline for (0..4) |c| inline for (0..4) |r| {
        try std.testing.expectApproxEqAbs(texp[c * 4 + r], t.data[c].v[r], 1e-6);
    };
    try expectMat4Approx(m4.mul(t), .{
        276, 304, 332, 360,
        304, 336, 368, 400,
        332, 368, 404, 440,
        360, 400, 440, 480,
    }, 1e-4);
    const mv = m4.mulVec(vec4.init(.{ 1, 2, 3, 1 }));
    try std.testing.expectApproxEqAbs(@as(f32, 51), mv.v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 58), mv.v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 65), mv.v[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 72), mv.v[3], 1e-6);
}

test "mat transforms" {
    const id = mat4.identity();
    try expectMat4Approx(id.translate(vec3.init(.{ 1, 2, 3 })), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 2, 3, 1,
    }, 1e-6);
    try expectMat4Approx(id.rotate(0.5, vec3.init(.{ 0, 1, 0 })), .{
        0.87758255, 0, -0.47942555, 0,
        0, 1, 0, 0,
        0.47942555, 0, 0.87758255, 0,
        0, 0, 0, 1,
    }, 1e-6);
    try expectMat4Approx(id.rotate(0.5, vec3.init(.{ 1, 0, 0 })), .{
        1, 0, 0, 0,
        0, 0.87758255, 0.47942555, 0,
        0, -0.47942555, 0.87758255, 0,
        0, 0, 0, 1,
    }, 1e-6);
    try expectMat4Approx(id.scale(vec3.init(.{ 2, 3, 4 })), .{
        2, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 4, 0,
        0, 0, 0, 1,
    }, 1e-6);
    const tt = id.translate(vec3.init(.{ 1, 2, 3 })).rotate(0.5, vec3.init(.{ 0, 1, 0 }));
    try expectMat4Approx(tt, .{
        0.87758255, 0, -0.47942555, 0,
        0, 1, 0, 0,
        0.47942555, 0, 0.87758255, 0,
        1, 2, 3, 1,
    }, 1e-6);
}

test "mat lookAt clip" {
    const eye = vec3.init(.{ 0, 0, 3 });
    const center = vec3.init(.{ 0, 0, 0 });
    const up = vec3.init(.{ 0, 1, 0 });
    try expectMat4Approx(mat.lookAt(eye, center, up), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -3, 1,
    }, 1e-6);
    try expectMat4Approx(mat.lookAtLH(eye, center, up), .{
        -1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 0,
        0, 0, 3, 1,
    }, 1e-6);
    const fovy = 0.78539816339744830961;
    const aspect = 16.0 / 9.0;
    try expectMat4Approx(mat.perspective(fovy, aspect, 0.1, 100.0), .{
        1.35799515, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1.002002, -1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveRH_ZO(fovy, aspect, 0.1, 100.0), .{
        1.35799515, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1.001001, -1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveLH_ZO(fovy, aspect, 0.1, 100.0), .{
        1.35799515, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, 1.001001, 1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.ortho(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -0.0200200193, 0,
        0, 0, -1.002002, 1,
    }, 1e-6);
    try expectMat4Approx(mat.orthoRH_ZO(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -0.0100100096, 0,
        0, 0, -0.00100100099, 1,
    }, 1e-6);
    try expectMat4Approx(mat.frustum(-1, 1, -1, 1, 0.1, 100.0), .{
        0.100000001, 0, 0, 0,
        0, 0.100000001, 0, 0,
        0, 0, -1.002002, -1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
}

test "mat clip variants" {
    const fovy = 0.78539816339744830961;
    const aspect = 16.0 / 9.0;
    const pz = [4]f32{ 1.35799515, 0, 0, 0 };
    const p1 = [4]f32{ 0, 2.41421342, 0, 0 };

    try expectMat4Approx(mat.ortho2D(-1, 1, -1, 1), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1,
    }, 1e-6);
    try expectMat4Approx(mat.orthoLH_ZO(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0.0100100096, 0,
        0, 0, -0.00100100099, 1,
    }, 1e-6);
    try expectMat4Approx(mat.orthoLH_NO(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0.0200200193, 0,
        0, 0, -1.002002, 1,
    }, 1e-6);
    try expectMat4Approx(mat.orthoRH_NO(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -0.0200200193, 0,
        0, 0, -1.002002, 1,
    }, 1e-6);
    try expectMat4Approx(mat.orthoZO(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -0.0100100096, 0,
        0, 0, -0.00100100099, 1,
    }, 1e-6);
    try expectMat4Approx(mat.orthoLH(-1, 1, -1, 1, 0.1, 100.0), .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0.0200200193, 0,
        0, 0, -1.002002, 1,
    }, 1e-6);

    try expectMat4Approx(mat.frustumLH_ZO(-1, 1, -1, 1, 0.1, 100.0), .{
        0.100000001, 0, 0, 0,
        0, 0.100000001, 0, 0,
        0, 0, 1.001001, 1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.frustumLH_NO(-1, 1, -1, 1, 0.1, 100.0), .{
        0.100000001, 0, 0, 0,
        0, 0.100000001, 0, 0,
        0, 0, 1.002002, 1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
    try expectMat4Approx(mat.frustumRH_ZO(-1, 1, -1, 1, 0.1, 100.0), .{
        0.100000001, 0, 0, 0,
        0, 0.100000001, 0, 0,
        0, 0, -1.001001, -1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.frustumRH_NO(-1, 1, -1, 1, 0.1, 100.0), .{
        0.100000001, 0, 0, 0,
        0, 0.100000001, 0, 0,
        0, 0, -1.002002, -1,
        0, 0, -0.2002002, 0,
    }, 1e-6);

    try expectMat4Approx(mat.perspectiveRH_NO(fovy, aspect, 0.1, 100.0), .{
        pz[0], pz[1], pz[2], pz[3],
        p1[0], p1[1], p1[2], p1[3],
        0, 0, -1.002002, -1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveLH_NO(fovy, aspect, 0.1, 100.0), .{
        pz[0], pz[1], pz[2], pz[3],
        p1[0], p1[1], p1[2], p1[3],
        0, 0, 1.002002, 1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveZO(fovy, aspect, 0.1, 100.0), .{
        pz[0], pz[1], pz[2], pz[3],
        p1[0], p1[1], p1[2], p1[3],
        0, 0, -1.001001, -1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveLH(fovy, aspect, 0.1, 100.0), .{
        pz[0], pz[1], pz[2], pz[3],
        p1[0], p1[1], p1[2], p1[3],
        0, 0, 1.002002, 1,
        0, 0, -0.2002002, 0,
    }, 1e-6);

    const fov = 0.78539816339744830961;
    try expectMat4Approx(mat.perspectiveFov(fov, 1280.0, 720.0, 0.1, 100.0), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1.002002, -1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveFovRH_ZO(fov, 1280.0, 720.0, 0.1, 100.0), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1.001001, -1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveFovLH_ZO(fov, 1280.0, 720.0, 0.1, 100.0), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, 1.001001, 1,
        0, 0, -0.1001001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveFovLH_NO(fov, 1280.0, 720.0, 0.1, 100.0), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, 1.002002, 1,
        0, 0, -0.2002002, 0,
    }, 1e-6);
    try expectMat4Approx(mat.perspectiveFovZO(fov, 1280.0, 720.0, 0.1, 100.0), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1.001001, -1,
        0, 0, -0.1001001, 0,
    }, 1e-6);

    try expectMat4Approx(mat.infinitePerspective(fovy, aspect, 0.1), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1, -1,
        0, 0, -0.200000003, 0,
    }, 1e-6);
    try expectMat4Approx(mat.infinitePerspectiveRH_ZO(fovy, aspect, 0.1), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -1, -1,
        0, 0, -0.100000001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.infinitePerspectiveLH_ZO(fovy, aspect, 0.1), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, 1, 1,
        0, 0, -0.100000001, 0,
    }, 1e-6);
    try expectMat4Approx(mat.infinitePerspectiveLH_NO(fovy, aspect, 0.1), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, 1, 1,
        0, 0, -0.200000003, 0,
    }, 1e-6);
    try expectMat4Approx(mat.tweakedInfinitePerspective(fovy, aspect, 0.1, std.math.floatEps(f32)), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -0.999999881, -1,
        0, 0, -0.199999988, 0,
    }, 1e-6);
    try expectMat4Approx(mat.tweakedInfinitePerspective(fovy, aspect, 0.1, 0.001), .{
        1.35799503, 0, 0, 0,
        0, 2.41421342, 0, 0,
        0, 0, -0.999000013, -1,
        0, 0, -0.199900001, 0,
    }, 1e-6);
}

test "mat projection" {
    const fovy = 0.78539816339744830961;
    const aspect = 16.0 / 9.0;
    const proj = mat.perspective(fovy, aspect, 0.1, 100.0);
    const model = mat4.identity().translate(vec3.init(.{ 1, 2, 3 }));
    const vp = vec4.init(.{ 0, 0, 800, 600 });
    const obj = vec3.init(.{ 1, 2, 3 });
    const win_no = mat.projectNO(obj, model, proj, vp);
    try std.testing.expectApproxEqAbs(@as(f32, 218.93396), win_no.v[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -182.842682), win_no.v[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.01768434), win_no.v[2], 1e-4);
    const win_zo = mat.projectZO(obj, model, proj, vp);
    try std.testing.expectApproxEqAbs(@as(f32, 218.93396), win_zo.v[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -182.842682), win_zo.v[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.03536868), win_zo.v[2], 1e-4);
    const un = mat.unProject(win_no, model, proj, vp);
    try std.testing.expectApproxEqAbs(@as(f32, 1.00000501), un.v[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0000155), un.v[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.00000572), un.v[2], 1e-5);
    const unz = mat.unProjectZO(win_no, model, proj, vp);
    try std.testing.expectApproxEqAbs(@as(f32, 3.255337), unz.v[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.51068544), unz.v[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9.76600361), unz.v[2], 1e-4);
    try expectMat4Approx(mat.pickMatrix(vec2.init(.{ 400, 300 }), vec2.init(.{ 100, 100 }), vp), .{
        8, 0, 0, 0,
        0, 6, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, 1e-6);
}

test "mat common relational inverse" {
    const A = mat4.diag(2).translate(vec3.init(.{ 1, 2, 3 }));
    const B = mat4.identity().rotate(0.5, vec3.init(.{ 0, 1, 0 }));
    try expectMat4Approx(A.abs(), .{
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        2, 4, 6, 2,
    }, 1e-6);
    try expectMat4Approx(A.mix(B, 0.25), .{
        1.71939564, 0, -0.119856387, 0,
        0, 1.75, 0, 0,
        0.119856387, 0, 1.71939564, 0,
        1.5, 3, 4.5, 1.75,
    }, 1e-6);
    try expectMat4Approx(A.mix(B, mat4.diag(2)), .{
        -0.2448349, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, -0.2448349, 0,
        2, 4, 6, 0,
    }, 1e-6);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, true, true }), A.equal(A).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, false, false }), A.equal(B).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, true, true }), A.notEqual(B).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, false, false }), A.equalEps(B, 0.1).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, true, true }), A.equalEps(B, vec4.init(.{ 0.1, 0.1, 100, 100 })).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, true, true }), A.notEqualEps(B, 0.1).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, false, false }), A.equalULP(B, 2).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, false, false }), A.equalULP(B, ivec4.init(.{ 2, 2, 10000, 10000 })).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, true, true }), A.notEqualULP(B, 2).v);

    const m3a = Mat(3, 3, f32).init(.{
        vec3.init(.{ 1, 2, 0 }),
        vec3.init(.{ 0, 1, 0 }),
        vec3.init(.{ 3, 4, 1 }),
    });
    const ai3 = mat.affineInverse3(m3a);
    try std.testing.expectApproxEqAbs(@as(f32, 1), ai3.data[0].v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2), ai3.data[0].v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3), ai3.data[2].v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), ai3.data[2].v[1], 1e-6);
    try expectMat4Approx(mat.affineInverse(A), .{
        0.5, 0, 0, 0,
        0, 0.5, 0, 0,
        0, 0, 0.5, 0,
        -1, -2, -3, 1,
    }, 1e-6);
    const it4 = mat.inverseTranspose(A);
    try expectMat4Approx(it4, .{
        0.5, -0.0, 0, -0.5,
        -0.0, 0.5, -0.0, -1,
        0, -0.0, 0.5, -1.5,
        -0.0, 0, -0.0, 0.5,
    }, 1e-6);
    const it3 = mat.inverseTranspose3(Mat(3, 3, f32).init(.{
        vec3.init(.{ 1, 2, 3 }),
        vec3.init(.{ 0, 1, 4 }),
        vec3.init(.{ 5, 6, 0 }),
    }));
    try std.testing.expectApproxEqAbs(@as(f32, -24), it3.data[0].v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20), it3.data[0].v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -5), it3.data[0].v[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4), it3.data[2].v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), it3.data[2].v[2], 1e-6);
}

test "mat comp mult outer" {
    const id = mat4.identity();
    const t = id.translate(vec3.init(.{ 1, 2, 3 }));
    try expectMat4Approx(t.matrixCompMult(mat4.diag(2)), .{
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 2,
    }, 1e-6);
    try expectMat4Approx(mat.outerProduct(vec3.init(.{ 1, 2, 3 }), vec3.init(.{ 4, 5, 6 })).toMat4(), .{
        4, 8, 12, 0,
        5, 10, 15, 0,
        6, 12, 18, 0,
        0, 0, 0, 1,
    }, 1e-6);
    try expectMat4Approx(mat.outerProduct(vec2.init(.{ 1, 2 }), vec3.init(.{ 4, 5, 6 })).toMat4(), .{
        4, 8, 0, 0,
        5, 10, 0, 0,
        6, 12, 1, 0,
        0, 0, 0, 1,
    }, 1e-6);
}

fn expectQuatApprox(actual: anytype, expected: [4]f32, tol: f32) !void {
    const T = @TypeOf(actual).value_type;
    const w: f32 = @floatCast(actual.w);
    const x: f32 = @floatCast(actual.x);
    const y: f32 = @floatCast(actual.y);
    const z: f32 = @floatCast(actual.z);
    _ = T;
    try std.testing.expectApproxEqAbs(expected[0], w, tol);
    try std.testing.expectApproxEqAbs(expected[1], x, tol);
    try std.testing.expectApproxEqAbs(expected[2], y, tol);
    try std.testing.expectApproxEqAbs(expected[3], z, tol);
}

// ---- quaternions ----

test "quat" {
    const q = quat.angleAxis(0.5, vec3.init(.{ 0, 1, 0 }));
    try expectQuatApprox(q, .{ 0.968912423, 0, 0.247403964, 0 }, 1e-6);
    try expectQuatApprox(q.mul(quat.angleAxis(0.3, vec3.init(.{ 1, 0, 0 }))), .{ 0.958032608, 0.144792467, 0.244625881, -0.0369715877 }, 1e-6);
    try expectQuatApprox(q.conjugate(), .{ 0.968912423, 0, -0.247403964, 0 }, 1e-6);
    try expectQuatApprox(q.inverse(), .{ 0.968912423, 0, -0.247403964, 0 }, 1e-6);
    const mv = q.mulVec3(vec3.init(.{ 1, 0, 0 }));
    try expectVecApprox(mv, vec3.init(.{ 0.87758255, 0, -0.47942555 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.958032608), q.dot(quat.angleAxis(0.3, vec3.init(.{ 1, 0, 0 }))), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), q.length(), 1e-6);
}

test "quat mat" {
    const q = quat.angleAxis(0.5, vec3.init(.{ 0, 1, 0 }));
    const m3 = quat.mat3_cast(q);
    inline for (0..3) |c| inline for (0..3) |r| {
        const exp = [3][3]f32{
            .{ 0.87758255, 0, -0.47942555 },
            .{ 0, 1, 0 },
            .{ 0.47942555, 0, 0.87758255 },
        };
        try std.testing.expectApproxEqAbs(exp[c][r], m3.data[c].v[r], 1e-6);
    };
    try expectMat4Approx(quat.mat4_cast(q), .{
        0.87758255, 0, -0.47942555, 0,
        0, 1, 0, 0,
        0.47942555, 0, 0.87758255, 0,
        0, 0, 0, 1,
    }, 1e-6);
    try expectQuatApprox(quat.quat_cast(m3), .{ 0.968912423, 0, 0.247403979, 0 }, 1e-6);
}

test "quat interp" {
    const q = quat.angleAxis(0.5, vec3.init(.{ 0, 1, 0 }));
    const id = Quat(f32).identity();
    try expectQuatApprox(quat.slerp(id, q, 0.25), .{ 0.998047471, 0, 0.0624593161, 0 }, 1e-6);
    try expectQuatApprox(quat.lerp(id, q, 0.25), .{ 0.992228091, 0, 0.0618509911, 0 }, 1e-6);
    try expectQuatApprox(quat.mix(id, q, 0.25), .{ 0.998047471, 0, 0.0624593161, 0 }, 1e-6);
}

test "quat angle axis euler" {
    const q = quat.angleAxis(0.5, vec3.init(.{ 0, 1, 0 }));
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), quat.angle(q), 1e-6);
    try expectVecApprox(quat.axis(q), vec3.init(.{ 0, 1, 0 }), 1e-6);
    const e = quat.eulerAngles(quat.angleAxis(0.5, vec3.init(.{ 1, 2, 3 })));
    try expectVecApprox(e, vec3.init(.{ 0.798036993, 0.633040369, 1.4500128 }), 1e-6);
    try expectQuatApprox(Quat(f32).fromEuler(vec3.init(.{ 0.5, 0.3, 0.2 })), .{ 0.956937492, 0.228948653, 0.168490946, 0.0588567853 }, 1e-6);
    const rv = quat.rotate(q, 0.5, vec3.init(.{ 1, 0, 0 })).mulVec3(vec3.init(.{ 1, 0, 0 }));
    try expectVecApprox(rv, vec3.init(.{ 0.87758255, 7.4505806e-09, -0.47942555 }), 1e-6);
    try expectQuatApprox(Quat(f32).init(2, 1, 0.5, 0.25).normalize(), .{ 0.867721856, 0.433860928, 0.216930464, 0.108465232 }, 1e-6);
}

test "quat exp log pow" {
    const q = quat.angleAxis(0.5, vec3.init(.{ 0, 1, 0 }));
    try expectQuatApprox(quat.exp(q), .{ 0.969551444, 0, 0.244887799, 0 }, 1e-6);
    try expectQuatApprox(quat.log(q), .{ 0, 0, 0.25, 0 }, 1e-6);
    try expectQuatApprox(quat.log(Quat(f32).init(-0.5, 0, 0, 0)), .{ -0.693147182, 3.14159274, 0, 0 }, 1e-6);
    try expectQuatApprox(quat.pow(q, 2.5), .{ 0.810963094, 0, 0.585097253, 0 }, 1e-6);
    try expectQuatApprox(quat.sqrt(q), .{ 0.992197692, 0, 0.12467473, 0 }, 1e-6);
    const q2 = quat.angleAxis(0.3, vec3.init(.{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, true, true }), quat.equal(q, q).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, true, false }), quat.equal(q, q2).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, true, true }), quat.equalEps(q, q2, 0.1).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, false, true }), quat.notEqual(q, q2).v);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ true, true, false, false }), quat.notEqualEps(q, q2, 0.1).v);
    const nan_q = Quat(f32).init(std.math.nan(f32), 0, 0, 0);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, false, true }), quat.isnan(nan_q).v);
    const inf_q = Quat(f32).init(std.math.inf(f32), 0, 0, 0);
    try std.testing.expectEqual(@as(@Vector(4, bool), .{ false, false, false, true }), quat.isinf(inf_q).v);
    try expectQuatApprox(quat.quatLookAtRH(vec3.init(.{ 0, 0, -1 }), vec3.init(.{ 0, 1, 0 })), .{ 1, 0, 0, 0 }, 1e-6);
    try expectQuatApprox(quat.quatLookAtLH(vec3.init(.{ 0, 0, -1 }), vec3.init(.{ 0, 1, 0 })), .{ 0, 0, 1, 0 }, 1e-6);
    try expectQuatApprox(quat.quatLookAt(vec3.init(.{ 0, 0, -1 }), vec3.init(.{ 0, 1, 0 })), .{ 1, 0, 0, 0 }, 1e-6);
    try expectQuatApprox(quat.slerpSpin(Quat(f32).init(1, 0, 0, 0), q, 0.25, 1), .{ 0.661560714, 0, 0.749891579, 0 }, 1e-6);
}

// ---- constructors ----

test "vec constructors" {
    const v4 = vec4.init(.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(f32, 2), v4.y());
    try std.testing.expectEqual(@as(f32, 4), v4.w());
    const v3 = vec3.init(.{ v4.xy(), 3.0 });
    try std.testing.expectEqual(@as(f32, 3), v3.z());
    const v3t = vec3.init(v4);
    try std.testing.expectEqual(@as(f32, 1), v3t.x());
    try expectVecApprox(vec3.init(5.0), vec3_(5, 5, 5), 1e-6);
    try expectVecApprox(v4.zyx(), vec3_(3, 2, 1), 1e-6);
    try expectVecApprox(v4.zw(), vec2.init(.{ 3, 4 }), 1e-6);
    const w = v4.add(1).sub(1).mul(2).div(2).neg().neg();
    try expectVecApprox(w, v4, 1e-6);
}
