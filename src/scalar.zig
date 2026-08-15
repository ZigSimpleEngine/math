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

// ---- ext/scalar_common ----

pub fn min3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    return min(min(x, y), z);
}

pub fn min4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    return min(min(x, y), min(z, w));
}

pub fn max3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    return max(max(x, y), z);
}

pub fn max4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    return max(max(x, y), max(z, w));
}

/// NaN-aware min (GLM fmin 2-arg).
pub fn fmin(x: anytype, y: anytype) @TypeOf(x, y) {
    if (isNan(x)) return y;
    if (isNan(y)) return x;
    return min(x, y);
}

pub fn fmin3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    if (isNan(x)) return fmin(y, z);
    if (isNan(y)) return fmin(x, z);
    if (isNan(z)) return min(x, y);
    return min3(x, y, z);
}

pub fn fmin4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    if (isNan(x)) return fmin3(y, z, w);
    if (isNan(y)) return min(x, fmin(z, w));
    if (isNan(z)) return fmin(min(x, y), w);
    if (isNan(w)) return min3(x, y, z);
    return min4(x, y, z, w);
}

/// NaN-aware max (GLM fmax 2-arg).
pub fn fmax(x: anytype, y: anytype) @TypeOf(x, y) {
    if (isNan(x)) return y;
    if (isNan(y)) return x;
    return max(x, y);
}

pub fn fmax3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    if (isNan(x)) return fmax(y, z);
    if (isNan(y)) return fmax(x, z);
    if (isNan(z)) return max(x, y);
    return max3(x, y, z);
}

pub fn fmax4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    if (isNan(x)) return fmax3(y, z, w);
    if (isNan(y)) return max(x, fmax(z, w));
    if (isNan(z)) return fmax(max(x, y), w);
    if (isNan(w)) return max3(x, y, z);
    return max4(x, y, z, w);
}

pub fn fclamp(x: anytype, min_val: anytype, max_val: anytype) @TypeOf(x, min_val, max_val) {
    return fmin(fmax(x, min_val), max_val);
}

pub fn clamp01(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return clamp(x, @as(T, 0), @as(T, 1));
}

pub fn repeat(x: anytype) @TypeOf(x) {
    return fract(x);
}

pub fn mirrorClamp(x: anytype) @TypeOf(x) {
    return fract(abs(x));
}

pub fn mirrorRepeat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const abs_x = abs(x);
    const clamp_v = mod(floor(abs_x), @as(T, 2));
    const floor_v = floor(abs_x);
    const rest = abs_x - floor_v;
    const mirror = clamp_v + rest;
    return mix(rest, @as(T, 1) - rest, mirror >= @as(T, 1));
}

/// GLM iround: int(x + 0.5) for x >= 0.
pub fn iround(x: anytype) i32 {
    const T = rtType(@TypeOf(x));
    return @as(i32, @intFromFloat(@as(T, x) + @as(T, 0.5)));
}

/// GLM uround: uint(x + 0.5) for x >= 0.
pub fn uround(x: anytype) u32 {
    const T = rtType(@TypeOf(x));
    return @as(u32, @intFromFloat(@as(T, x) + @as(T, 0.5)));
}

// ---- ext/scalar_reciprocal ----

pub fn sec(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / cos(@as(T, x));
}

pub fn csc(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / sin(@as(T, x));
}

pub fn cot(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    const pi_over_2: T = cast(T, 3.1415926535897932384626433832795 / 2.0);
    return tan(pi_over_2 - @as(T, x));
}

pub fn asec(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return acos(@as(T, 1) / @as(T, x));
}

pub fn acsc(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return asin(@as(T, 1) / @as(T, x));
}

pub fn acot(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    const pi_over_2: T = cast(T, 3.1415926535897932384626433832795 / 2.0);
    return pi_over_2 - atan(@as(T, x));
}

pub fn sech(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / cosh(@as(T, x));
}

pub fn csch(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / sinh(@as(T, x));
}

pub fn coth(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return cosh(@as(T, x)) / sinh(@as(T, x));
}

pub fn asech(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return acosh(@as(T, 1) / @as(T, x));
}

pub fn acsch(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return asinh(@as(T, 1) / @as(T, x));
}

pub fn acoth(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return atanh(@as(T, 1) / @as(T, x));
}

// ---- ext/scalar_relational + gtc/epsilon ----

pub fn equalEps(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) <= eps;
}

pub fn notEqualEps(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) > eps;
}

pub fn epsilonEqual(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) < eps;
}

pub fn epsilonNotEqual(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) >= eps;
}

const FltInt = struct {
    fn asI32(f: f32) i32 {
        return @as(i32, @bitCast(f));
    }
    fn asI64(d: f64) i64 {
        return @as(i64, @bitCast(d));
    }
};

/// GLM equal(x, y, MaxULPs): bit-exact ULP comparison.
pub fn equalULP(x: anytype, y: anytype, max_ulps: anytype) bool {
    const T = @TypeOf(x);
    if (T == f64) {
        const a: i64 = FltInt.asI64(x);
        const b: i64 = FltInt.asI64(y);
        if ((a < 0) != (b < 0)) return false;
        return abs(a - b) <= @as(i64, max_ulps);
    } else {
        const a: i32 = FltInt.asI32(x);
        const b: i32 = FltInt.asI32(y);
        if ((a < 0) != (b < 0)) return false;
        return abs(a - b) <= @as(i32, max_ulps);
    }
}

pub fn notEqualULP(x: anytype, y: anytype, max_ulps: anytype) bool {
    return !equalULP(x, y, max_ulps);
}

// ---- ext/scalar_ulp ----

/// GLM nextFloat(x): next representable float toward +inf (std::nextafter to max).
pub fn nextFloat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return std.math.nextAfter(x, std.math.inf(T));
}

pub fn nextFloatN(x: anytype, ulps: i32) @TypeOf(x) {
    var temp = x;
    var i: i32 = 0;
    while (i < ulps) : (i += 1) temp = nextFloat(temp);
    return temp;
}

/// GLM prevFloat(x): next representable float toward -inf (std::nextafter to -max).
pub fn prevFloat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return std.math.nextAfter(x, -std.math.inf(T));
}

pub fn prevFloatN(x: anytype, ulps: i32) @TypeOf(x) {
    var temp = x;
    var i: i32 = 0;
    while (i < ulps) : (i += 1) temp = prevFloat(temp);
    return temp;
}

/// GLM floatDistance(x, y): abs(bit-pattern distance) — note GLM returns abs(a.i - b.i)
/// without the sign check used by equalULP.
pub fn floatDistance(x: anytype, y: anytype) i64 {
    const T = @TypeOf(x, y);
    if (T == f64) {
        return abs(FltInt.asI64(x) - FltInt.asI64(y));
    } else {
        const a: f32 = @floatCast(x);
        const b: f32 = @floatCast(y);
        return abs(@as(i64, FltInt.asI32(a)) - @as(i64, FltInt.asI32(b)));
    }
}

// ---- bit-casts (func_common) ----

pub fn floatBitsToInt(v: f32) i32 {
    return @as(i32, @bitCast(v));
}

pub fn floatBitsToUint(v: f32) u32 {
    return @as(u32, @bitCast(v));
}

pub fn intBitsToFloat(v: i32) f32 {
    return @as(f32, @bitCast(v));
}

pub fn uintBitsToFloat(v: u32) f32 {
    return @as(f32, @bitCast(v));
}

// ---- ext/scalar_integer + gtc/round ----

pub fn isPowerOfTwo(value: anytype) bool {
    const T = @TypeOf(value);
    const result = if (isSigned(T)) abs(value) else value;
    return (result & (result - 1)) == 0;
}

/// C++-semantics integer remainder (truncated, matches GLM's `%`).
pub fn remInt(a: anytype, b: anytype) @TypeOf(a, b) {
    return a % b;
}

pub fn isMultiple(value: anytype, multiple: anytype) bool {
    return value % multiple == 0;
}

pub fn ceilPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const bits: comptime_int = @typeInfo(T).int.bits;
    const sign_val = sign(value);
    var v = abs(value) -% 1;
    comptime var shift: u6 = 1;
    inline while (shift < bits) : (shift <<= 1) {
        v |= v >> shift;
    }
    return (v +% 1) * sign_val;
}

pub fn floorPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    return @as(T, 1) << @intCast(findMSB(value));
}

pub fn roundPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    const prev = @as(T, 1) << @intCast(findMSB(value));
    const next = prev << 1;
    return if ((next - value) < (value - prev)) next else prev;
}

pub fn nextPowerOfTwo(value: anytype) @TypeOf(value) {
    return ceilPowerOfTwo(value);
}

pub fn prevPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    return @as(T, 1) << @intCast(findMSB(value));
}

pub fn nextMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    return ceilMultiple(source, multiple);
}

pub fn prevMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    return floorMultiple(source, multiple);
}

pub fn ceilMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    const T = @TypeOf(source, multiple);
    if (comptime isFloat(T)) {
        if (source > @as(T, 0)) {
            return source + (multiple - @rem(source, multiple));
        } else {
            return source + @rem(-source, multiple);
        }
    } else if (comptime isSigned(T)) {
        if (source > @as(T, 0)) {
            const tmp = source - 1;
            return tmp + (multiple - remInt(tmp, multiple));
        } else {
            return source + remInt(-source, multiple);
        }
    } else {
        const tmp = source -% 1;
        return tmp + (multiple - remInt(tmp, multiple));
    }
}

pub fn floorMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    const T = @TypeOf(source, multiple);
    if (comptime isFloat(T)) {
        if (source >= @as(T, 0)) {
            return source - @rem(source, multiple);
        } else {
            return source - @rem(source, multiple) - multiple;
        }
    } else if (source >= @as(T, 0)) {
        return source - remInt(source, multiple);
    } else {
        const tmp = source + 1;
        return tmp - remInt(tmp, multiple) - multiple;
    }
}

pub fn roundMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    const T = @TypeOf(source, multiple);
    if (comptime isFloat(T)) {
        if (source >= @as(T, 0)) {
            return source - @rem(source, multiple);
        } else {
            const tmp = source + @as(T, 1);
            return tmp - @rem(tmp, multiple) - multiple;
        }
    } else if (source >= @as(T, 0)) {
        return source - remInt(source, multiple);
    } else {
        const tmp = source + 1;
        return tmp - remInt(tmp, multiple) - multiple;
    }
}

/// GLM findNSB: position of the most significant group of `count` set bits.
pub fn findNSB(x: anytype, significant_bit_count: i32) i32 {
    const T = @TypeOf(x);
    const bits: comptime_int = @typeInfo(T).int.bits;
    if (bitCount(x) < significant_bit_count) return -1;
    const one: T = 1;
    var bit_pos: i32 = 0;
    var key = x;
    var n_bit_count = significant_bit_count;
    var bit_step: i32 = @intCast(bits / 2);
    while (key > one) {
        const mask = (@as(T, 1) << @intCast(bit_step)) - 1;
        const current_key = key & mask;
        const current_bit_count = bitCount(current_key);
        if (n_bit_count > current_bit_count) {
            n_bit_count -= current_bit_count;
            bit_pos += bit_step;
            key >>= @intCast(bit_step);
        } else {
            key = key & mask;
        }
        bit_step >>= 1;
        if (bit_step == 0) break;
    }
    return bit_pos;
}
