//! Scalar math helpers — GLM-equivalent implementations.
//! Used internally by Vec/Mat/Quat methods and exposed for convenience.

const std = @import("std");

pub fn isFloat(comptime T: type) bool {
    return @typeInfo(T) == .float;
}

pub fn isInt(comptime T: type) bool {
    return @typeInfo(T) == .int;
}

pub fn isSigned(comptime T: type) bool {
    return isInt(T) and @typeInfo(T).int.signedness == .signed;
}

pub fn isNumber(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .comptime_int, .comptime_float, .int, .float, .bool => true,
        else => false,
    };
}

/// GLM's `float_t<T>`: double stays double, everything else float.
pub fn floatType(comptime T: type) type {
    return if (T == f64 or T == f16) T else f32;
}

/// Runtime equivalent of comptime numerics (comptime float/int -> f32).
pub fn rtType(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .comptime_int, .comptime_float => f32,
        else => T,
    };
}

/// Cast any numeric value to T (implicit-conversion semantics of C++).
pub fn cast(comptime T: type, x: anytype) T {
    const U = @TypeOf(x);
    if (U == T) return x;
    return switch (@typeInfo(U)) {
        .comptime_int, .comptime_float => if (T == bool) x != 0 else @as(T, x),
        .int => switch (@typeInfo(T)) {
            .bool => x != 0,
            .int => @intCast(x),
            .float => @floatFromInt(x),
            else => @compileError("cannot cast integer to " ++ @typeName(T)),
        },
        .float => switch (@typeInfo(T)) {
            .bool => x != 0,
            .float => @floatCast(x),
            .int => @intFromFloat(@trunc(x)),
            else => @compileError("cannot cast float to " ++ @typeName(T)),
        },
        .bool => switch (@typeInfo(T)) {
            .bool => x,
            .int => @intFromBool(x),
            else => @compileError("cannot cast bool to " ++ @typeName(T)),
        },
        else => @compileError("cannot cast " ++ @typeName(U) ++ " to " ++ @typeName(T)),
    };
}

// ---- common ----

pub fn abs(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return switch (@typeInfo(T)) {
        .comptime_int, .comptime_float => @abs(x),
        .float => @abs(x),
        // @abs on function parameters is broken in Zig 0.16.0; manual form matches GLM.
        .int => if (comptime isSigned(T)) (if (x < 0) -%x else x) else x,
        else => @compileError("abs: unsupported type " ++ @typeName(T)),
    };
}

pub fn sign(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const zero: T = 0;
    return @as(T, @intFromBool(x > zero)) - @as(T, @intFromBool(x < zero));
}

pub fn floor(x: anytype) @TypeOf(x) {
    return @floor(x);
}

pub fn ceil(x: anytype) @TypeOf(x) {
    return @ceil(x);
}

pub fn round(x: anytype) @TypeOf(x) {
    return @round(x);
}

/// GLM's roundEven: rounds half-way cases to the nearest even integer.
pub fn roundEven(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const integer: i32 = @intFromFloat(@trunc(x));
    const integer_part: T = @floatFromInt(integer);
    const fractional_part = fract(x);
    if (fractional_part > @as(T, 0.5) or fractional_part < @as(T, 0.5)) {
        return round(x);
    } else if (@mod(integer, 2) == 0) {
        return integer_part;
    } else if (x <= @as(T, 0)) {
        return integer_part - @as(T, 1);
    } else {
        return integer_part + @as(T, 1);
    }
}

pub fn trunc(x: anytype) @TypeOf(x) {
    return @trunc(x);
}

pub fn fract(x: anytype) @TypeOf(x) {
    return x - floor(x);
}

pub fn mod(x: anytype, y: anytype) @TypeOf(x, y) {
    if (comptime isFloat(@TypeOf(x))) return x - y * floor(x / y);
    return x % y;
}

pub fn min(x: anytype, y: anytype) @TypeOf(x, y) {
    return if (y < x) y else x;
}

pub fn max(x: anytype, y: anytype) @TypeOf(x, y) {
    return if (x < y) y else x;
}

pub fn clamp(x: anytype, min_val: anytype, max_val: anytype) @TypeOf(x, min_val, max_val) {
    return min(max(x, min_val), max_val);
}

pub fn mix(x: anytype, y: anytype, a: anytype) @TypeOf(x, y) {
    if (comptime @TypeOf(a) == bool) return if (a) y else x;
    return x * (1 - a) + y * a;
}

pub fn step(edge: anytype, x: anytype) @TypeOf(edge, x) {
    return mix(@as(@TypeOf(edge, x), 1), @as(@TypeOf(edge, x), 0), x < edge);
}

pub fn smoothstep(edge0: anytype, edge1: anytype, x: anytype) @TypeOf(edge0, edge1, x) {
    const T = @TypeOf(edge0, edge1, x);
    const t = clamp((x - edge0) / (edge1 - edge0), @as(T, 0), @as(T, 1));
    return t * t * (@as(T, 3) - @as(T, 2) * t);
}

/// GLM's fma uses a fused multiply-add for scalars, a*b + c otherwise.
pub fn fma(a: anytype, b: anytype, c: anytype) @TypeOf(a, b, c) {
    return @mulAdd(@TypeOf(a, b, c), a, b, c);
}

pub fn modf(x: anytype) struct { fract: @TypeOf(x), integral: @TypeOf(x) } {
    const T = @TypeOf(x);
    const integral: T = trunc(x);
    return .{ .fract = x - integral, .integral = integral };
}

pub fn frexp(x: anytype) struct { significand: floatType(@TypeOf(x)), exponent: i32 } {
    const F = floatType(@TypeOf(x));
    const r = std.math.frexp(@as(F, x));
    return .{ .significand = r.significand, .exponent = r.exponent };
}

pub fn ldexp(x: anytype, e: i32) floatType(@TypeOf(x)) {
    const F = floatType(@TypeOf(x));
    return std.math.ldexp(@as(F, x), e);
}

pub fn isNan(x: anytype) bool {
    return std.math.isNan(x);
}

pub fn isInf(x: anytype) bool {
    return std.math.isInf(x);
}

// ---- trigonometric ----

pub fn radians(x: anytype) @TypeOf(x) {
    return x * @as(@TypeOf(x), 3.14159265358979323846264338327950288) / @as(@TypeOf(x), 180);
}

pub fn degrees(x: anytype) @TypeOf(x) {
    return x * @as(@TypeOf(x), 180) / @as(@TypeOf(x), 3.14159265358979323846264338327950288);
}

pub fn sin(x: anytype) @TypeOf(x) {
    return @sin(x);
}

pub fn cos(x: anytype) @TypeOf(x) {
    return @cos(x);
}

pub fn tan(x: anytype) @TypeOf(x) {
    return @tan(x);
}

pub fn asin(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.asin(@as(T, x));
}

pub fn acos(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.acos(@as(T, x));
}

pub fn atan(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.atan(@as(T, x));
}

pub fn atan2(y: anytype, x: anytype) rtType(@TypeOf(y, x)) {
    const T = rtType(@TypeOf(y, x));
    return std.math.atan2(@as(T, y), @as(T, x));
}

pub fn sinh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.sinh(@as(T, x));
}

pub fn cosh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.cosh(@as(T, x));
}

pub fn tanh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.tanh(@as(T, x));
}

pub fn asinh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.asinh(@as(T, x));
}

pub fn acosh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.acosh(@as(T, x));
}

pub fn atanh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.atanh(@as(T, x));
}

// ---- exponential ----

pub fn pow(x: anytype, y: anytype) rtType(@TypeOf(x, y)) {
    const T = rtType(@TypeOf(x, y));
    return std.math.pow(T, x, y);
}

pub fn exp(x: anytype) @TypeOf(x) {
    return @exp(x);
}

pub fn exp2(x: anytype) @TypeOf(x) {
    return @exp2(x);
}

pub fn log(x: anytype) @TypeOf(x) {
    return @log(x);
}

pub fn log2(x: anytype) @TypeOf(x) {
    return @log2(x);
}

pub fn sqrt(x: anytype) @TypeOf(x) {
    return @sqrt(x);
}

pub fn inversesqrt(x: anytype) @TypeOf(x) {
    return @as(@TypeOf(x), 1) / @sqrt(x);
}

// ---- bit / integer ----

pub fn bitCount(x: anytype) i32 {
    return @as(i32, @popCount(x));
}

/// Index of the least significant 1 bit, or -1 if x == 0 (GLM semantics).
pub fn findLSB(x: anytype) i32 {
    const T = @TypeOf(x);
    if (x == 0) return -1;
    const U = std.meta.Int(.unsigned, @typeInfo(T).int.bits);
    return @as(i32, @ctz(@as(U, @bitCast(x))));
}

/// Index of the most significant 1 bit, or -1 if x == 0 (GLM semantics).
/// For signed inputs the bit pattern is used (arithmetic-fill like GLM).
pub fn findMSB(x: anytype) i32 {
    const T = @TypeOf(x);
    const bits: comptime_int = @typeInfo(T).int.bits;
    if (x == 0) return -1;
    var v = x;
    comptime var shift: u7 = 1;
    inline while (shift < 64) : (shift <<= 1) {
        if (bits > shift) v |= v >> shift;
    }
    const U = std.meta.Int(.unsigned, bits);
    const zeros: i32 = @as(i32, @popCount(~@as(U, @bitCast(v))));
    return @as(i32, @intCast(bits - 1)) - zeros;
}

pub fn bitfieldExtract(value: anytype, offset: i32, bits: i32) @TypeOf(value) {
    const T = @TypeOf(value);
    const U = std.meta.Int(.unsigned, @typeInfo(T).int.bits);
    const uv: U = @bitCast(value);
    const mask: U = (@as(U, 1) << @intCast(bits)) -% 1;
    const extracted = (uv >> @intCast(offset)) & mask;
    return @bitCast(extracted);
}

pub fn bitfieldInsert(base: anytype, insert: anytype, offset: i32, bits: i32) @TypeOf(base, insert) {
    const T = @TypeOf(base, insert);
    const U = std.meta.Int(.unsigned, @typeInfo(T).int.bits);
    const ub: U = @bitCast(base);
    const ui: U = @bitCast(insert);
    const mask: U = (@as(U, 1) << @intCast(bits)) -% 1;
    const shifted_mask = mask << @intCast(offset);
    return @bitCast((ub & ~shifted_mask) | ((ui << @intCast(offset)) & shifted_mask));
}

pub fn bitfieldReverse(value: anytype) @TypeOf(value) {
    return @bitReverse(value);
}

pub const AddCarryResult = struct { sum: u32, carry: u32 };

pub fn uaddCarry(x: anytype, y: anytype) struct { sum: @TypeOf(x, y), carry: @TypeOf(x, y) } {
    const T = @TypeOf(x, y);
    const r = @addWithOverflow(x, y);
    return .{ .sum = r[0], .carry = if (r[1]) @as(T, 1) else @as(T, 0) };
}

pub fn usubBorrow(x: anytype, y: anytype) struct { diff: @TypeOf(x, y), borrow: @TypeOf(x, y) } {
    const T = @TypeOf(x, y);
    const r = @subWithOverflow(x, y);
    return .{ .diff = r[0], .borrow = if (r[1]) @as(T, 1) else @as(T, 0) };
}

// ---- reductions ----

pub fn compAdd(x: anytype) @TypeOf(x) {
    return x;
}
