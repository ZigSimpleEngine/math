//! Scalar math helpers — single-value versions of the GLM math API.
//!
//! Every function accepts plain Zig numerics (`f32`, `f64`, integers, and
//! comptime literals) and reproduces GLM 1.1.0 semantics where they differ
//! from the obvious one: NaN handling (`fmin` vs `min`), rounding modes
//! (`roundEven`, `iround`), ULP comparison, and truncated integer division.
//! Vec/Mat/Quat delegate per-component math here, and the functions are
//! exposed for direct use in application code.

const std = @import("std");

/// Returns `true` if `T` is a floating-point type (f16/f32/f64/f80/f128).
/// Use in comptime conditionals to dispatch float vs integer behavior.
pub fn isFloat(comptime T: type) bool {
    return @typeInfo(T) == .float;
}

/// Returns `true` if `T` is a signed or unsigned integer type.
pub fn isInt(comptime T: type) bool {
    return @typeInfo(T) == .int;
}

/// Returns `true` if `T` is a *signed* integer type.
pub fn isSigned(comptime T: type) bool {
    return isInt(T) and @typeInfo(T).int.signedness == .signed;
}

/// Returns `true` if `T` can be passed to the arithmetic functions here:
/// comptime literals, integers, floats and bools.
pub fn isNumber(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .comptime_int, .comptime_float, .int, .float, .bool => true,
        else => false,
    };
}

/// Resolve the "working float" type for a GLM-style computation:
/// `f64` and `f16` map to themselves, everything else (including comptime
/// floats and integers) maps to `f32`. Use it when a function must pick a
/// concrete float storage type regardless of the input literal.
pub fn floatType(comptime T: type) type {
    return if (T == f64 or T == f16) T else f32;
}

/// Runtime equivalent of a comptime numeric type: `comptime_int` and
/// `comptime_float` become `f32` (GLM's default `float`), everything else
/// stays unchanged. Most return types in this module are computed with it.
pub fn rtType(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .comptime_int, .comptime_float => f32,
        else => T,
    };
}

/// Convert `x` to `T` following C++ implicit-conversion rules: bool <-> int
/// <-> float round-trips, floats narrow to integers by truncation, comptime
/// values convert at compile time. Use `scalar.cast(T, v)` wherever the GLM
/// reference relies on an implicit conversion of a literal or variable.
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

/// Absolute value: floats, comptime values and signed integers. Integers use
/// the manual `(x < 0 ? -x : x)` form instead of `@abs` to work around a
/// Zig 0.16 bug with `@abs` on function parameters.
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

/// Sign of `x`: `-1` if negative, `+1` if positive, `0` if zero. Use to
/// extract a direction from a signed value, e.g. `sign(velocity)`.
pub fn sign(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const zero: T = 0;
    return @as(T, @intFromBool(x > zero)) - @as(T, @intFromBool(x < zero));
}

/// Largest integer not greater than `x` — rounds toward -inf.
/// Use to quantize a continuous value down to whole steps.
pub fn floor(x: anytype) @TypeOf(x) {
    return @floor(x);
}

/// Smallest integer not less than `x` — rounds toward +inf.
/// Use to quantize a continuous value up to whole steps.
pub fn ceil(x: anytype) @TypeOf(x) {
    return @ceil(x);
}

/// Rounds to the nearest integer; half-way cases go away from zero
/// (C `round` semantics). For unbiased rounding use `roundEven`.
pub fn round(x: anytype) @TypeOf(x) {
    return @round(x);
}

/// Rounds to the nearest integer; half-way cases go to the nearest *even*
/// integer (GLSL `roundEven`). Unlike `round` this has no directional bias,
/// so it is the right choice for repeated accumulation/quantization.
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

/// Removes the fractional part — rounds toward zero.
pub fn trunc(x: anytype) @TypeOf(x) {
    return @trunc(x);
}

/// Fractional part `x - floor(x)`, always in `[0, 1)`.
/// Use for seamless tiling (texture coordinates, looping time, waveform
/// phase): `fract(t)` repeats the ramp `0..1` with period 1.
pub fn fract(x: anytype) @TypeOf(x) {
    return x - floor(x);
}

/// For floats: `x - y*floor(x/y)` — result is in `[0, |y|)` and has the sign
/// of `y` (so `mod(5.3, 2) == 1.3`, `mod(-5.3, 2) == 0.7`).
/// For integers: truncated remainder, sign follows `x`.
pub fn mod(x: anytype, y: anytype) @TypeOf(x, y) {
    if (comptime isFloat(@TypeOf(x))) return x - y * floor(x / y);
    return x % y;
}

/// Smaller of `x` and `y`. NaN propagates (returns NaN) — use `fmin` when
/// NaN values should be ignored instead of poisoning the result.
pub fn min(x: anytype, y: anytype) @TypeOf(x, y) {
    return if (y < x) y else x;
}

/// Larger of `x` and `y`. NaN propagates — use `fmax` when NaNs should be
/// ignored.
pub fn max(x: anytype, y: anytype) @TypeOf(x, y) {
    return if (x < y) y else x;
}

/// Constrain `x` to the range `[min_val, max_val]`.
/// The canonical call is `clamp(v, 0, 1)` — normalizing an arbitrary value
/// into an interpolation factor, alpha or UV coordinate.
pub fn clamp(x: anytype, min_val: anytype, max_val: anytype) @TypeOf(x, min_val, max_val) {
    return min(max(x, min_val), max_val);
}

/// Linear interpolation: `x*(1-a) + y*a`, `a` in `[0, 1]` (values outside are
/// extrapolated). If `a` is a `bool`, returns `y` when true and `x` otherwise
/// — the GLSL `select` idiom.
pub fn mix(x: anytype, y: anytype, a: anytype) @TypeOf(x, y) {
    if (comptime @TypeOf(a) == bool) return if (a) y else x;
    return x * (1 - a) + y * a;
}

/// Step function: `0` if `x < edge`, `1` otherwise.
/// Turns a continuous value into a binary signal, e.g. gate/on-off control.
pub fn step(edge: anytype, x: anytype) @TypeOf(edge, x) {
    return mix(@as(@TypeOf(edge, x), 1), @as(@TypeOf(edge, x), 0), x < edge);
}

/// Hermite smoothing `t*t*(3 - 2t)` with `t = clamp((x - edge0) /
/// (edge1 - edge0), 0, 1)`. Produces a smooth 0→1 transition with zero
/// slope at both ends — use for easing, anti-aliased edges, soft falloff.
pub fn smoothstep(edge0: anytype, edge1: anytype, x: anytype) @TypeOf(edge0, edge1, x) {
    const T = @TypeOf(edge0, edge1, x);
    const t = clamp((x - edge0) / (edge1 - edge0), @as(T, 0), @as(T, 1));
    return t * t * (@as(T, 3) - @as(T, 2) * t);
}

/// Fused multiply-add `a*b + c` with a single rounding step (IEEE `fma`).
/// More accurate than `a*b + c`; use in sensitive accumulations (matrices,
/// sums of many terms) to reduce rounding error.
pub fn fma(a: anytype, b: anytype, c: anytype) @TypeOf(a, b, c) {
    return @mulAdd(@TypeOf(a, b, c), a, b, c);
}

/// Split `x` into its integral and fractional parts; both parts keep the
/// sign of `x` (unlike `fract`, which is always positive).
pub fn modf(x: anytype) struct { fract: @TypeOf(x), integral: @TypeOf(x) } {
    const T = @TypeOf(x);
    const integral: T = trunc(x);
    return .{ .fract = x - integral, .integral = integral };
}

/// Split `x` into a significand in `[0.5, 1)` and an integer exponent such
/// that `x == significand * 2^exponent`. Use to decompose a float's
/// magnitude or to detect exponent ranges (e.g. for log-space work).
pub fn frexp(x: anytype) struct { significand: floatType(@TypeOf(x)), exponent: i32 } {
    const F = floatType(@TypeOf(x));
    const r = std.math.frexp(@as(F, x));
    return .{ .significand = r.significand, .exponent = r.exponent };
}

/// Inverse of `frexp`: reconstructs `x * 2^e` without rounding through
/// exponent arithmetic. Use to scale by powers of two exactly.
pub fn ldexp(x: anytype, e: i32) floatType(@TypeOf(x)) {
    const F = floatType(@TypeOf(x));
    return std.math.ldexp(@as(F, x), e);
}

/// Returns `true` if `x` is NaN (not-a-number). Use to guard values that may
/// come out of invalid math (0/0, asin out of domain).
pub fn isNan(x: anytype) bool {
    return std.math.isNan(x);
}

/// Returns `true` if `x` is +inf or -inf.
pub fn isInf(x: anytype) bool {
    return std.math.isInf(x);
}

// ---- trigonometric ----

/// Convert degrees to radians (`x * pi/180`). Use when constructing angles
/// from human-readable values: `radians(90) == pi/2`.
pub fn radians(x: anytype) @TypeOf(x) {
    return x * @as(@TypeOf(x), 3.14159265358979323846264338327950288) / @as(@TypeOf(x), 180);
}

/// Convert radians to degrees (`x * 180/pi`). Use for display/debug output.
pub fn degrees(x: anytype) @TypeOf(x) {
    return x * @as(@TypeOf(x), 180) / @as(@TypeOf(x), 3.14159265358979323846264338327950288);
}

/// Sine of `x` (radians). For oscillating signals prefer keeping `x` small
/// (modulo by 2π) to preserve precision.
pub fn sin(x: anytype) @TypeOf(x) {
    return @sin(x);
}

/// Cosine of `x` (radians).
pub fn cos(x: anytype) @TypeOf(x) {
    return @cos(x);
}

/// Tangent of `x` (radians). Singular at π/2 ± kπ.
pub fn tan(x: anytype) @TypeOf(x) {
    return @tan(x);
}

/// Arcsine, result in `[-pi/2, pi/2]`. Returns NaN for inputs outside
/// `[-1, 1]` — clamp the argument first when the source is not guaranteed
/// to be normalized.
pub fn asin(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.asin(@as(T, x));
}

/// Arccosine, result in `[0, pi]`. NaN for inputs outside `[-1, 1]`.
pub fn acos(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.acos(@as(T, x));
}

/// Arctangent, result in `[-pi/2, pi/2]`. For a full 360° angle from a
/// vector use `atan2(y, x)` instead.
pub fn atan(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.atan(@as(T, x));
}

/// Quadrant-correct arctangent of `y/x`, result in `(-pi, pi]`.
/// This is the function to compute the heading of a vector:
/// `atan2(dy, dx)` gives the full 360° angle, `atan2(y, x) == pi/2` for
/// `(x, y) == (0, 1)`.
pub fn atan2(y: anytype, x: anytype) rtType(@TypeOf(y, x)) {
    const T = rtType(@TypeOf(y, x));
    return std.math.atan2(@as(T, y), @as(T, x));
}

/// Hyperbolic sine.
pub fn sinh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.sinh(@as(T, x));
}

/// Hyperbolic cosine. Use for catenary-style curves and for the stable
/// `cosh(x) + sinh(x) == exp(x)` identity.
pub fn cosh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.cosh(@as(T, x));
}

/// Hyperbolic tangent, result in `(-1, 1)`. Use as a smooth saturating
/// (sigmoid-like) response curve for velocities/joystick inputs.
pub fn tanh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.tanh(@as(T, x));
}

/// Inverse hyperbolic sine, defined for all reals.
pub fn asinh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.asinh(@as(T, x));
}

/// Inverse hyperbolic cosine, defined for `x >= 1`.
pub fn acosh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.acosh(@as(T, x));
}

/// Inverse hyperbolic tangent, defined for `|x| < 1`.
pub fn atanh(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return std.math.atanh(@as(T, x));
}

// ---- exponential ----

/// `x` raised to the power `y`. Returns NaN for a negative base with a
/// non-integer exponent. Typical uses: easing curves `pow(t, 2)`,
/// exponential falloff, gamma correction.
pub fn pow(x: anytype, y: anytype) rtType(@TypeOf(x, y)) {
    const T = rtType(@TypeOf(x, y));
    return std.math.pow(T, x, y);
}

/// Euler's number raised to `x`. Base function for natural-log math;
/// use in decay curves like `exp(-t * rate)`.
pub fn exp(x: anytype) @TypeOf(x) {
    return @exp(x);
}

/// `2^x`. Cheaper than `pow(2, x)`; use for power-of-two scaling in
/// shader-style code.
pub fn exp2(x: anytype) @TypeOf(x) {
    return @exp2(x);
}

/// Natural logarithm, defined for `x > 0`.
pub fn log(x: anytype) @TypeOf(x) {
    return @log(x);
}

/// Base-2 logarithm. Use to convert ratios into octaves/semitones or to
/// find the exponent of a power of two.
pub fn log2(x: anytype) @TypeOf(x) {
    return @log2(x);
}

/// Square root, defined for `x >= 0`. Use for distances/lengths; the
/// dedicated `length()`/`normalize()` paths are faster.
pub fn sqrt(x: anytype) @TypeOf(x) {
    return @sqrt(x);
}

/// Reciprocal square root `1/sqrt(x)`. Cheaper than `1/sqrt(x)` on most
/// hardware; the classic way to normalize vectors when you have the squared
/// length already.
pub fn inversesqrt(x: anytype) @TypeOf(x) {
    return @as(@TypeOf(x), 1) / @sqrt(x);
}

// ---- bit / integer ----

/// Population count: number of 1 bits in `x`. Use for Hamming distance,
/// bitboard tricks or validating sparse flag masks.
pub fn bitCount(x: anytype) i32 {
    return @as(i32, @popCount(x));
}

/// Index of the least significant set bit (0-based), or `-1` when `x == 0`.
/// Example: `findLSB(0b01010000) == 4`. Use to extract the lowest set flag
/// or to fast-divide by the trailing power of two.
pub fn findLSB(x: anytype) i32 {
    const T = @TypeOf(x);
    if (x == 0) return -1;
    const U = std.meta.Int(.unsigned, @typeInfo(T).int.bits);
    return @as(i32, @ctz(@as(U, @bitCast(x))));
}

/// Index of the most significant set bit (0-based), or `-1` when `x == 0`.
/// For signed inputs the two's-complement bit pattern is used, so
/// `findMSB(-1) == 31`. Use `findMSB(x)` as a cheap `floor(log2(x))` for
/// positive integers.
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

/// Extract `bits` bits starting at `offset` (bit 0 = LSB), as GLSL
/// `bitfieldExtract`. Example: `bitfieldExtract(0b11011010, 3, 4) == 0b1011`.
/// Use for decoding packed bitfields (flag groups, color channels).
pub fn bitfieldExtract(value: anytype, offset: i32, bits: i32) @TypeOf(value) {
    const T = @TypeOf(value);
    const U = std.meta.Int(.unsigned, @typeInfo(T).int.bits);
    const uv: U = @bitCast(value);
    const mask: U = (@as(U, 1) << @intCast(bits)) -% 1;
    const extracted = (uv >> @intCast(offset)) & mask;
    return @bitCast(extracted);
}

/// Replace `bits` bits of `base` (starting at `offset`) with the low `bits`
/// bits of `insert`; all other bits of `base` are kept.
/// Use for packing several small fields into one integer.
pub fn bitfieldInsert(base: anytype, insert: anytype, offset: i32, bits: i32) @TypeOf(base, insert) {
    const T = @TypeOf(base, insert);
    const U = std.meta.Int(.unsigned, @typeInfo(T).int.bits);
    const ub: U = @bitCast(base);
    const ui: U = @bitCast(insert);
    const mask: U = (@as(U, 1) << @intCast(bits)) -% 1;
    const shifted_mask = mask << @intCast(offset);
    return @bitCast((ub & ~shifted_mask) | ((ui << @intCast(offset)) & shifted_mask));
}

/// Reverse the order of the bits of `x` (LSB becomes MSB). Use for
/// symmetric hashing of small bit patterns.
pub fn bitfieldReverse(value: anytype) @TypeOf(value) {
    return @bitReverse(value);
}

pub const AddCarryResult = struct { sum: u32, carry: u32 };

/// Unsigned addition that reports overflow: returns `sum` (wrapping) and
/// `carry == 1` if `x + y` exceeded the type's range. Use for multi-word
/// (big number) arithmetic chains.
pub fn uaddCarry(x: anytype, y: anytype) struct { sum: @TypeOf(x, y), carry: @TypeOf(x, y) } {
    const T = @TypeOf(x, y);
    const r = @addWithOverflow(x, y);
    return .{ .sum = r[0], .carry = if (r[1]) @as(T, 1) else @as(T, 0) };
}

/// Unsigned subtraction that reports underflow: returns `diff` (wrapping)
/// and `borrow == 1` if `y > x`. Use for multi-word arithmetic chains.
pub fn usubBorrow(x: anytype, y: anytype) struct { diff: @TypeOf(x, y), borrow: @TypeOf(x, y) } {
    const T = @TypeOf(x, y);
    const r = @subWithOverflow(x, y);
    return .{ .diff = r[0], .borrow = if (r[1]) @as(T, 1) else @as(T, 0) };
}

// ---- reductions ----

/// GLM `compAdd`: sum of all components of a vector. For a scalar this is
/// the identity — the function exists for API symmetry with the vector
/// version.
pub fn compAdd(x: anytype) @TypeOf(x) {
    return x;
}

// ---- ext/scalar_common ----

/// Smallest of three values — avoids nested `min` calls.
pub fn min3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    return min(min(x, y), z);
}

/// Smallest of four values.
pub fn min4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    return min(min(x, y), min(z, w));
}

/// Largest of three values.
pub fn max3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    return max(max(x, y), z);
}

/// Largest of four values.
pub fn max4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    return max(max(x, y), max(z, w));
}

/// NaN-tolerant minimum (IEEE `fmin`): if one argument is NaN the other is
/// returned, so a single bad value cannot poison the result. Use for
/// clamping in pipelines where NaN may legitimately appear.
pub fn fmin(x: anytype, y: anytype) @TypeOf(x, y) {
    if (isNan(x)) return y;
    if (isNan(y)) return x;
    return min(x, y);
}

/// NaN-tolerant minimum of three values.
pub fn fmin3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    if (isNan(x)) return fmin(y, z);
    if (isNan(y)) return fmin(x, z);
    if (isNan(z)) return min(x, y);
    return min3(x, y, z);
}

/// NaN-tolerant minimum of four values.
pub fn fmin4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    if (isNan(x)) return fmin3(y, z, w);
    if (isNan(y)) return min(x, fmin(z, w));
    if (isNan(z)) return fmin(min(x, y), w);
    if (isNan(w)) return min3(x, y, z);
    return min4(x, y, z, w);
}

/// NaN-tolerant maximum (IEEE `fmax`): NaN arguments are ignored.
pub fn fmax(x: anytype, y: anytype) @TypeOf(x, y) {
    if (isNan(x)) return y;
    if (isNan(y)) return x;
    return max(x, y);
}

/// NaN-tolerant maximum of three values.
pub fn fmax3(x: anytype, y: anytype, z: anytype) @TypeOf(x, y, z) {
    if (isNan(x)) return fmax(y, z);
    if (isNan(y)) return fmax(x, z);
    if (isNan(z)) return max(x, y);
    return max3(x, y, z);
}

/// NaN-tolerant maximum of four values.
pub fn fmax4(x: anytype, y: anytype, z: anytype, w: anytype) @TypeOf(x, y, z, w) {
    if (isNan(x)) return fmax3(y, z, w);
    if (isNan(y)) return max(x, fmax(z, w));
    if (isNan(z)) return fmax(max(x, y), w);
    if (isNan(w)) return max3(x, y, z);
    return max4(x, y, z, w);
}

/// NaN-tolerant clamp: `fmin(fmax(x, lo), hi)`. Unlike `clamp` a NaN input
/// does not leak into the result.
pub fn fclamp(x: anytype, min_val: anytype, max_val: anytype) @TypeOf(x, min_val, max_val) {
    return fmin(fmax(x, min_val), max_val);
}

/// Clamp into `[0, 1]`. The standard way to turn any value into a valid
/// interpolation factor, alpha or normalized coordinate.
pub fn clamp01(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return clamp(x, @as(T, 0), @as(T, 1));
}

/// Sawtooth wave in `[0, 1)` with period 1 (same as `fract`). Use to loop
/// a time/coordinate signal: `repeat(t)` gives an endlessly repeating
/// 0→1 ramp.
pub fn repeat(x: anytype) @TypeOf(x) {
    return fract(x);
}

/// Sawtooth of the absolute value: `mirrorClamp(-0.3) == 0.3`. The result
/// stays in `[0, 1)` for any input; use on signed signals that must remain
/// non-negative.
pub fn mirrorClamp(x: anytype) @TypeOf(x) {
    return fract(abs(x));
}

/// Triangle wave in `[0, 1]`: for `x` growing, the output cycles
/// `0 → 1 → 0 → 1 ...`. Use for ping-pong animation loops and mirrored
/// texture tiling.
pub fn mirrorRepeat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const abs_x = abs(x);
    const clamp_v = mod(floor(abs_x), @as(T, 2));
    const floor_v = floor(abs_x);
    const rest = abs_x - floor_v;
    const mirror = clamp_v + rest;
    return mix(rest, @as(T, 1) - rest, mirror >= @as(T, 1));
}

/// Round to nearest `i32` using `int(x + 0.5)` (truncating). Note this is
/// asymmetric for negatives (`iround(-1.5) == -1`), matching GLSL `iround`.
/// Use when the result must be fed into an integer API.
pub fn iround(x: anytype) i32 {
    const T = rtType(@TypeOf(x));
    return @as(i32, @intFromFloat(@as(T, x) + @as(T, 0.5)));
}

/// Round to nearest `u32` via `uint(x + 0.5)`; the input is expected to be
/// non-negative.
pub fn uround(x: anytype) u32 {
    const T = rtType(@TypeOf(x));
    return @as(u32, @intFromFloat(@as(T, x) + @as(T, 0.5)));
}

// ---- ext/scalar_reciprocal ----

/// Secant `1/cos(x)`. Singular at `pi/2 + k*pi`.
pub fn sec(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / cos(@as(T, x));
}

/// Cosecant `1/sin(x)`. Singular at multiples of `pi`.
pub fn csc(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / sin(@as(T, x));
}

/// Cotangent `cos(x)/sin(x)`, computed as `tan(pi/2 - x)` for numerical
/// stability around small angles. Singular at multiples of `pi`.
pub fn cot(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    const pi_over_2: T = cast(T, 3.1415926535897932384626433832795 / 2.0);
    return tan(pi_over_2 - @as(T, x));
}

/// Inverse secant `acos(1/x)`, defined for `|x| >= 1`.
pub fn asec(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return acos(@as(T, 1) / @as(T, x));
}

/// Inverse cosecant `asin(1/x)`, defined for `|x| >= 1`.
pub fn acsc(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return asin(@as(T, 1) / @as(T, x));
}

/// Inverse cotangent `pi/2 - atan(x)`, result in `(0, pi)`.
pub fn acot(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    const pi_over_2: T = cast(T, 3.1415926535897932384626433832795 / 2.0);
    return pi_over_2 - atan(@as(T, x));
}

/// Hyperbolic secant `1/cosh(x)` — a bell-shaped curve that decays to
/// zero at large |x|.
pub fn sech(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / cosh(@as(T, x));
}

/// Hyperbolic cosecant `1/sinh(x)`. Singular at `x == 0`.
pub fn csch(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return @as(T, 1) / sinh(@as(T, x));
}

/// Hyperbolic cotangent `cosh(x)/sinh(x)`. Singular at `x == 0`, tends to
/// ±1 at infinity.
pub fn coth(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return cosh(@as(T, x)) / sinh(@as(T, x));
}

/// Inverse hyperbolic secant `acosh(1/x)`, defined for `x` in `(0, 1]`.
pub fn asech(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return acosh(@as(T, 1) / @as(T, x));
}

/// Inverse hyperbolic cosecant `asinh(1/x)`, defined for `x != 0`.
pub fn acsch(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return asinh(@as(T, 1) / @as(T, x));
}

/// Inverse hyperbolic cotangent `atanh(1/x)`, defined for `|x| > 1`.
pub fn acoth(x: anytype) rtType(@TypeOf(x)) {
    const T = rtType(@TypeOf(x));
    return atanh(@as(T, 1) / @as(T, x));
}

// ---- ext/scalar_relational + gtc/epsilon ----

/// Tolerance comparison: true if `|x - y| <= eps`. Use instead of `==`
/// whenever `x`/`y` are results of floating-point arithmetic.
pub fn equalEps(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) <= eps;
}

/// Complement of `equalEps`: true if `|x - y| > eps`.
pub fn notEqualEps(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) > eps;
}

/// Strict tolerance comparison: true if `|x - y| < eps`. The epsilon
/// variant of GLM's gtc/epsilon module — useful for "is this inside a
/// tolerance ball" checks.
pub fn epsilonEqual(x: anytype, y: anytype, eps: anytype) bool {
    return abs(x - y) < eps;
}

/// Complement of `epsilonEqual`: true if `|x - y| >= eps`.
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

/// ULP-based float comparison: true if `x` and `y` are at most `max_ulps`
/// representable steps apart (bit patterns compared as signed integers,
/// with the sign bit handled separately). Unlike an epsilon test this works
/// at *any* magnitude — `1.0` vs the float just below it is 1 ULP, the same
/// as `1e30` vs its neighbor. Caveat: `+0.0` and `-0.0` compare unequal.
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

/// Complement of `equalULP`.
pub fn notEqualULP(x: anytype, y: anytype, max_ulps: anytype) bool {
    return !equalULP(x, y, max_ulps);
}

// ---- ext/scalar_ulp ----

/// Next representable float strictly greater than `x` (walks toward +inf).
/// Use to iterate a float lattice or to compute "1 ULP above" a value.
pub fn nextFloat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return std.math.nextAfter(x, std.math.inf(T));
}

/// Apply `nextFloat` `ulps` times — moves `ulps` steps up the float
/// lattice in one call.
pub fn nextFloatN(x: anytype, ulps: i32) @TypeOf(x) {
    var temp = x;
    var i: i32 = 0;
    while (i < ulps) : (i += 1) temp = nextFloat(temp);
    return temp;
}

/// Next representable float strictly smaller than `x` (walks toward -inf).
/// Use to compute "1 ULP below" a value.
pub fn prevFloat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return std.math.nextAfter(x, -std.math.inf(T));
}

/// Apply `prevFloat` `ulps` times — moves `ulps` steps down the float
/// lattice in one call.
pub fn prevFloatN(x: anytype, ulps: i32) @TypeOf(x) {
    var temp = x;
    var i: i32 = 0;
    while (i < ulps) : (i += 1) temp = prevFloat(temp);
    return temp;
}

/// Absolute distance between `x` and `y` measured in ULPs (number of
/// representable floats between them). Note: this variant intentionally
/// skips the sign-bit check of `equalULP`, so `floatDistance(-1, 1)` is a
/// huge number. Use for a magnitude-independent error metric.
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

/// Reinterpret the bits of an `f32` as `i32` (no conversion).
/// Use to sort/compare floats by bit pattern, or to store data in the
/// mantissa bits. Inverse: `intBitsToFloat`.
pub fn floatBitsToInt(v: f32) i32 {
    return @as(i32, @bitCast(v));
}

/// Reinterpret the bits of an `f32` as `u32`. Same idea as
/// `floatBitsToInt` with an unsigned view.
pub fn floatBitsToUint(v: f32) u32 {
    return @as(u32, @bitCast(v));
}

/// Reconstruct an `f32` from an `i32` bit pattern; the inverse of
/// `floatBitsToInt`.
pub fn intBitsToFloat(v: i32) f32 {
    return @as(f32, @bitCast(v));
}

/// Reconstruct an `f32` from a `u32` bit pattern; the inverse of
/// `floatBitsToUint`.
pub fn uintBitsToFloat(v: u32) f32 {
    return @as(f32, @bitCast(v));
}

// ---- ext/scalar_integer + gtc/round ----

/// True if `value` is a power of two (`1, 2, 4, ...`); the sign is ignored
/// for signed types, and `0` also returns true (bit trick, matches GLM).
/// Use to validate sizes before bit-shift scaling.
pub fn isPowerOfTwo(value: anytype) bool {
    const T = @TypeOf(value);
    const result = if (isSigned(T)) abs(value) else value;
    return (result & (result - 1)) == 0;
}

/// Truncated integer remainder (C++ `%` semantics: sign follows the
/// dividend). Zig's `%` already behaves this way for ints; the function is
/// a named alias used for clarity inside the round/multiple helpers.
pub fn remInt(a: anytype, b: anytype) @TypeOf(a, b) {
    return a % b;
}

/// True if `value` is divisible by `multiple` (not limited to powers of
/// two, unlike `isPowerOfTwo`).
pub fn isMultiple(value: anytype, multiple: anytype) bool {
    return value % multiple == 0;
}

/// Smallest power of two `>= |value|`, preserving the sign:
/// `ceilPowerOfTwo(5) == 8`, `ceilPowerOfTwo(-5) == -8`. Use for buffer
/// sizes, texture dimensions and data alignment.
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

/// Largest power of two `<= value`: `floorPowerOfTwo(5) == 4`. For values
/// that are already powers of two the input is returned unchanged.
pub fn floorPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    return @as(T, 1) << @intCast(findMSB(value));
}

/// Nearest power of two; ties resolve upward (`roundPowerOfTwo(5) == 4`,
/// `roundPowerOfTwo(7) == 8`, `roundPowerOfTwo(6) == 8`). Use when you need
/// a power-of-two that "best" approximates a size.
pub fn roundPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    const prev = @as(T, 1) << @intCast(findMSB(value));
    const next = prev << 1;
    return if ((next - value) < (value - prev)) next else prev;
}

/// Alias for `ceilPowerOfTwo`.
pub fn nextPowerOfTwo(value: anytype) @TypeOf(value) {
    return ceilPowerOfTwo(value);
}

/// Alias for `floorPowerOfTwo`.
pub fn prevPowerOfTwo(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    return @as(T, 1) << @intCast(findMSB(value));
}

/// Alias for `ceilMultiple`.
pub fn nextMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    return ceilMultiple(source, multiple);
}

/// Alias for `floorMultiple`.
pub fn prevMultiple(source: anytype, multiple: anytype) @TypeOf(source, multiple) {
    return floorMultiple(source, multiple);
}

/// Smallest multiple of `multiple` that is `>= source`:
/// `ceilMultiple(14, 8) == 16`, `ceilMultiple(16, 8) == 16`. Handles
/// floats and signed/unsigned ints. Use to round up to an alignment or a
/// fixed stride: `ceilMultiple(byte_count, 16)`.
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

/// Largest multiple of `multiple` that is `<= source`:
/// `floorMultiple(14, 8) == 8`, `floorMultiple(15, 8) == 8`. Use to snap a
/// value down to a grid.
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

/// GLM gtc/round `roundMultiple`: for non-negative inputs this equals
/// `floorMultiple`; negative inputs use a compensating +1 trick so the
/// result is always a multiple of `multiple`.
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

/// Index of the start of the most significant run of `significant_bit_count`
/// consecutive set bits, or `-1` if `x` has fewer set bits in total.
/// Use for locating the topmost dense region of a bitmask.
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
