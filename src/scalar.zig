//! Scalar math helpers — single-value versions of the GLM math API.
//!
//! Every function accepts plain Zig numerics (`f32`, `f64`, integers, and
//! comptime literals) and reproduces GLM 1.1.0 semantics where they differ
//! from the obvious one: NaN handling (`fmin` vs `min`), rounding modes
//! (`roundEven`, `iround`), ULP comparison, and truncated integer division.
//! Vec/Mat/Quat delegate per-component math here, and the functions are
//! exposed for direct use in application code.

const std = @import("std");

/// Returns `true` if `scalar_type` is a floating-point type (f16/f32/f64/f80/f128).
/// Use in comptime conditionals to dispatch float vs integer behavior.
pub fn isFloat(comptime scalar_type: type) bool {
    return @typeInfo(scalar_type) == .float;
}

/// Returns `true` if `scalar_type` is a signed or unsigned integer type.
pub fn isInt(comptime scalar_type: type) bool {
    return @typeInfo(scalar_type) == .int;
}

/// Returns `true` if `scalar_type` is a *signed* integer type.
pub fn isSigned(comptime scalar_type: type) bool {
    return isInt(scalar_type) and @typeInfo(scalar_type).int.signedness == .signed;
}

/// Returns `true` if `scalar_type` can be passed to the arithmetic functions here:
/// comptime literals, integers, floats and bools.
pub fn isNumber(comptime scalar_type: type) bool {
    return switch (@typeInfo(scalar_type)) {
        .comptime_int, .comptime_float, .int, .float, .bool => true,
        else => false,
    };
}

/// Resolve the "working float" type for a GLM-style computation:
/// `f64` and `f16` map to themselves, everything else (including comptime
/// floats and integers) maps to `f32`. Use it when a function must pick a
/// concrete float storage type regardless of the input literal.
pub fn floatType(comptime scalar_type: type) type {
    return if (scalar_type == f64 or scalar_type == f16) scalar_type else f32;
}

/// Runtime equivalent of a comptime numeric type: `comptime_int` and
/// `comptime_float` become `f32` (GLM's default `float`), everything else
/// stays unchanged. Most return types in this module are computed with it.
pub fn rtType(comptime scalar_type: type) type {
    return switch (@typeInfo(scalar_type)) {
        .comptime_int, .comptime_float => f32,
        else => scalar_type,
    };
}

/// Convert `value` to `scalar_type` following C++ implicit-conversion rules: bool <-> int
/// <-> float round-trips, floats narrow to integers by truncation, comptime
/// values convert at compile time. Use `scalar.cast(scalar_type, value)` wherever the GLM
/// reference relies on an implicit conversion of a literal or variable.
pub fn cast(comptime scalar_type: type, value: anytype) scalar_type {
    const U = @TypeOf(value);
    if (U == scalar_type) return value;
    return switch (@typeInfo(U)) {
        .comptime_int, .comptime_float => if (scalar_type == bool) value != 0 else @as(scalar_type, value),
        .int => switch (@typeInfo(scalar_type)) {
            .bool => value != 0,
            .int => @intCast(value),
            .float => @floatFromInt(value),
            else => @compileError("cannot cast integer to " ++ @typeName(scalar_type)),
        },
        .float => switch (@typeInfo(scalar_type)) {
            .bool => value != 0,
            .float => @floatCast(value),
            .int => @intFromFloat(@trunc(value)),
            else => @compileError("cannot cast float to " ++ @typeName(scalar_type)),
        },
        .bool => switch (@typeInfo(scalar_type)) {
            .bool => value,
            .int => @intFromBool(value),
            else => @compileError("cannot cast bool to " ++ @typeName(scalar_type)),
        },
        else => @compileError("cannot cast " ++ @typeName(U) ++ " to " ++ @typeName(scalar_type)),
    };
}

// ---- common ----

/// Absolute value: floats, comptime values and signed integers. Integers use
/// the manual `(x < 0 ? -x : x)` form instead of `@abs` to work around a
/// Zig 0.16 bug with `@abs` on function parameters.
pub fn abs(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    return switch (@typeInfo(scalar_type)) {
        .comptime_int, .comptime_float => @abs(value),
        .float => @abs(value),
        // @abs on function parameters is broken in Zig 0.16.0; manual form matches GLM.
        .int => if (comptime isSigned(scalar_type)) (if (value < 0) -%value else value) else value,
        else => @compileError("abs: unsupported type " ++ @typeName(scalar_type)),
    };
}

/// Sign of `value`: `-1` if negative, `+1` if positive, `0` if zero. Use to
/// extract a direction from a signed value, e.g. `sign(velocity)`.
pub fn sign(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    const zero: scalar_type = 0;
    return @as(scalar_type, @intFromBool(value > zero)) - @as(scalar_type, @intFromBool(value < zero));
}

/// Largest integer not greater than `value` — rounds toward -inf.
/// Use to quantize a continuous value down to whole steps.
pub fn floor(value: anytype) @TypeOf(value) {
    return @floor(value);
}

/// Smallest integer not less than `value` — rounds toward +inf.
/// Use to quantize a continuous value up to whole steps.
pub fn ceil(value: anytype) @TypeOf(value) {
    return @ceil(value);
}

/// Rounds to the nearest integer; half-way cases go away from zero
/// (C `round` semantics). For unbiased rounding use `roundEven`.
pub fn round(value: anytype) @TypeOf(value) {
    return @round(value);
}

/// Rounds to the nearest integer; half-way cases go to the nearest *even*
/// integer (GLSL `roundEven`). Unlike `round` this has no directional bias,
/// so it is the right choice for repeated accumulation/quantization.
pub fn roundEven(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    const integer: i32 = @intFromFloat(@trunc(value));
    const integer_part: scalar_type = @floatFromInt(integer);
    const fractional_part = fract(value);
    if (fractional_part > @as(scalar_type, 0.5) or fractional_part < @as(scalar_type, 0.5)) {
        return round(value);
    } else if (@mod(integer, 2) == 0) {
        return integer_part;
    } else if (value <= @as(scalar_type, 0)) {
        return integer_part - @as(scalar_type, 1);
    } else {
        return integer_part + @as(scalar_type, 1);
    }
}

/// Removes the fractional part — rounds toward zero.
pub fn trunc(value: anytype) @TypeOf(value) {
    return @trunc(value);
}

/// Fractional part `value - floor(value)`, always in `[0, 1)`.
/// Use for seamless tiling (texture coordinates, looping time, waveform
/// phase): `fract(t)` repeats the ramp `0..1` with period 1.
pub fn fract(value: anytype) @TypeOf(value) {
    return value - floor(value);
}

/// For floats: `dividend - divisor*floor(dividend/divisor)` — result is in `[0, |divisor|)` and has the sign
/// of `divisor` (so `mod(5.3, 2) == 1.3`, `mod(-5.3, 2) == 0.7`).
/// For integers: truncated remainder, sign follows `dividend`.
pub fn mod(dividend: anytype, divisor: anytype) @TypeOf(dividend, divisor) {
    if (comptime isFloat(@TypeOf(dividend))) return dividend - divisor * floor(dividend / divisor);
    return dividend % divisor;
}

/// Smaller of `left_hand_side` and `right_hand_side`. NaN propagates (returns NaN) — use `fmin` when
/// NaN values should be ignored instead of poisoning the result.
pub fn min(left_hand_side: anytype, right_hand_side: anytype) @TypeOf(left_hand_side, right_hand_side) {
    return if (right_hand_side < left_hand_side) right_hand_side else left_hand_side;
}

/// Larger of `left_hand_side` and `right_hand_side`. NaN propagates — use `fmax` when NaNs should be
/// ignored.
pub fn max(left_hand_side: anytype, right_hand_side: anytype) @TypeOf(left_hand_side, right_hand_side) {
    return if (left_hand_side < right_hand_side) right_hand_side else left_hand_side;
}

/// Constrain `value` to the range `[min_val, max_val]`.
/// The canonical call is `clamp(v, 0, 1)` — normalizing an arbitrary value
/// into an interpolation factor, alpha or UV coordinate.
pub fn clamp(value: anytype, min_val: anytype, max_val: anytype) @TypeOf(value, min_val, max_val) {
    return min(max(value, min_val), max_val);
}

/// Linear interpolation: `from*(1-factor) + to*factor`, `factor` in `[0, 1]` (values outside are
/// extrapolated). If `factor` is a `bool`, returns `to` when true and `from` otherwise
/// — the GLSL `select` idiom.
pub fn mix(from: anytype, to: anytype, factor: anytype) @TypeOf(from, to) {
    if (comptime @TypeOf(factor) == bool) return if (factor) to else from;
    return from * (1 - factor) + to * factor;
}

/// Step function: `0` if `value < edge`, `1` otherwise.
/// Turns a continuous value into a binary signal, e.g. gate/on-off control.
pub fn step(edge: anytype, value: anytype) @TypeOf(edge, value) {
    return mix(@as(@TypeOf(edge, value), 1), @as(@TypeOf(edge, value), 0), value < edge);
}

/// Hermite smoothing `t*t*(3 - 2t)` with `t = clamp((x - edge0) /
/// (edge1 - edge0), 0, 1)`. Produces a smooth 0→1 transition with zero
/// slope at both ends — use for easing, anti-aliased edges, soft falloff.
pub fn smoothstep(edge0: anytype, edge1: anytype, value: anytype) @TypeOf(edge0, edge1, value) {
    const scalar_type = @TypeOf(edge0, edge1, value);
    const t = clamp((value - edge0) / (edge1 - edge0), @as(scalar_type, 0), @as(scalar_type, 1));
    return t * t * (@as(scalar_type, 3) - @as(scalar_type, 2) * t);
}

/// Fused multiply-add `left_hand_side*right_hand_side + addend` with a single rounding step (IEEE `fma`).
/// More accurate than `left_hand_side*right_hand_side + addend`; use in sensitive accumulations (matrices,
/// sums of many terms) to reduce rounding error.
pub fn fma(left_hand_side: anytype, right_hand_side: anytype, addend: anytype) @TypeOf(left_hand_side, right_hand_side, addend) {
    return @mulAdd(@TypeOf(left_hand_side, right_hand_side, addend), left_hand_side, right_hand_side, addend);
}

/// Split `value` into its integral and fractional parts; both parts keep the
/// sign of `value` (unlike `fract`, which is always positive).
pub fn modf(value: anytype) extern struct { fract: @TypeOf(value), integral: @TypeOf(value) } {
    const scalar_type = @TypeOf(value);
    const integral: scalar_type = trunc(value);
    return .{ .fract = value - integral, .integral = integral };
}

/// Split `value` into a significand in `[0.5, 1)` and an integer exponent such
/// that `value == significand * 2^exponent`. Use to decompose a float's
/// magnitude or to detect exponent ranges (e.g. for log-space work).
pub fn frexp(value: anytype) extern struct { significand: floatType(@TypeOf(value)), exponent: i32 } {
    const F = floatType(@TypeOf(value));
    const r = std.math.frexp(@as(F, value));
    return .{ .significand = r.significand, .exponent = r.exponent };
}

/// Inverse of `frexp`: reconstructs `value * 2^exponent` without rounding through
/// exponent arithmetic. Use to scale by powers of two exactly.
pub fn ldexp(value: anytype, exponent: i32) floatType(@TypeOf(value)) {
    const F = floatType(@TypeOf(value));
    return std.math.ldexp(@as(F, value), exponent);
}

/// Returns `true` if `value` is NaN (not-a-number). Use to guard values that may
/// come out of invalid math (0/0, asin out of domain).
pub fn isNan(value: anytype) bool {
    return std.math.isNan(value);
}

/// Returns `true` if `value` is +inf or -inf.
pub fn isInf(value: anytype) bool {
    return std.math.isInf(value);
}

// ---- trigonometric ----

/// Convert degrees to radians (`angle_degrees * pi/180`). Use when constructing angles
/// from human-readable values: `radians(90) == pi/2`.
pub fn radians(angle_degrees: anytype) @TypeOf(angle_degrees) {
    return angle_degrees * @as(@TypeOf(angle_degrees), 3.14159265358979323846264338327950288) / @as(@TypeOf(angle_degrees), 180);
}

/// Convert radians to degrees (`angle_radians * 180/pi`). Use for display/debug output.
pub fn degrees(angle_radians: anytype) @TypeOf(angle_radians) {
    return angle_radians * @as(@TypeOf(angle_radians), 180) / @as(@TypeOf(angle_radians), 3.14159265358979323846264338327950288);
}

/// Sine of `value` (radians). For oscillating signals prefer keeping `value` small
/// (modulo by 2π) to preserve precision.
pub fn sin(value: anytype) @TypeOf(value) {
    return @sin(value);
}

/// Cosine of `value` (radians).
pub fn cos(value: anytype) @TypeOf(value) {
    return @cos(value);
}

/// Tangent of `value` (radians). Singular at π/2 ± kπ.
pub fn tan(value: anytype) @TypeOf(value) {
    return @tan(value);
}

/// Arcsine, result in `[-pi/2, pi/2]`. Returns NaN for inputs outside
/// `[-1, 1]` — clamp the argument first when the source is not guaranteed
/// to be normalized.
pub fn asin(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.asin(@as(scalar_type, value));
}

/// Arccosine, result in `[0, pi]`. NaN for inputs outside `[-1, 1]`.
pub fn acos(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.acos(@as(scalar_type, value));
}

/// Arctangent, result in `[-pi/2, pi/2]`. For a full 360° angle from a
/// vector use `atan2(y, x)` instead.
pub fn atan(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.atan(@as(scalar_type, value));
}

/// Quadrant-correct arctangent of `y/x`, result in `(-pi, pi]`.
/// This is the function to compute the heading of a vector:
/// `atan2(dy, dx)` gives the full 360° angle, `atan2(y, x) == pi/2` for
/// `(x, y) == (0, 1)`.
pub fn atan2(y: anytype, x: anytype) rtType(@TypeOf(y, x)) {
    const scalar_type = rtType(@TypeOf(y, x));
    return std.math.atan2(@as(scalar_type, y), @as(scalar_type, x));
}

/// Hyperbolic sine.
pub fn sinh(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.sinh(@as(scalar_type, value));
}

/// Hyperbolic cosine. Use for catenary-style curves and for the stable
/// `cosh(x) + sinh(x) == exp(x)` identity.
pub fn cosh(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.cosh(@as(scalar_type, value));
}

/// Hyperbolic tangent, result in `(-1, 1)`. Use as a smooth saturating
/// (sigmoid-like) response curve for velocities/joystick inputs.
pub fn tanh(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.tanh(@as(scalar_type, value));
}

/// Inverse hyperbolic sine, defined for all reals.
pub fn asinh(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.asinh(@as(scalar_type, value));
}

/// Inverse hyperbolic cosine, defined for `value >= 1`.
pub fn acosh(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.acosh(@as(scalar_type, value));
}

/// Inverse hyperbolic tangent, defined for `|value| < 1`.
pub fn atanh(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return std.math.atanh(@as(scalar_type, value));
}

// ---- exponential ----

/// `base` raised to the power `exponent`. Returns NaN for a negative base with a
/// non-integer exponent. Typical uses: easing curves `pow(t, 2)`,
/// exponential falloff, gamma correction.
pub fn pow(base: anytype, exponent: anytype) rtType(@TypeOf(base, exponent)) {
    const scalar_type = rtType(@TypeOf(base, exponent));
    return std.math.pow(scalar_type, base, exponent);
}

/// Euler's number raised to `value`. Base function for natural-log math;
/// use in decay curves like `exp(-t * rate)`.
pub fn exp(value: anytype) @TypeOf(value) {
    return @exp(value);
}

/// `2^x`. Cheaper than `pow(2, x)`; use for power-of-two scaling in
/// shader-style code.
pub fn exp2(value: anytype) @TypeOf(value) {
    return @exp2(value);
}

/// Natural logarithm, defined for `value > 0`.
pub fn log(value: anytype) @TypeOf(value) {
    return @log(value);
}

/// Base-2 logarithm. Use to convert ratios into octaves/semitones or to
/// find the exponent of a power of two.
pub fn log2(value: anytype) @TypeOf(value) {
    return @log2(value);
}

/// Square root, defined for `value >= 0`. Use for distances and lengths.
pub fn sqrt(value: anytype) @TypeOf(value) {
    return @sqrt(value);
}

/// Reciprocal square root: `1/sqrt(value)`, computed as a division by `sqrt`.
/// The classic way to normalize a vector when you have its squared length
/// already.
pub fn inversesqrt(value: anytype) @TypeOf(value) {
    return @as(@TypeOf(value), 1) / @sqrt(value);
}

// ---- bit / integer ----

/// Population count: number of 1 bits in `value`. Use for Hamming distance,
/// bitboard tricks or validating sparse flag masks.
pub fn bitCount(value: anytype) i32 {
    return @as(i32, @popCount(value));
}

/// Index of the least significant set bit (0-based), or `-1` when `value == 0`.
/// Example: `findLSB(0b01010000) == 4`. Use to extract the lowest set flag
/// or to fast-divide by the trailing power of two.
pub fn findLSB(value: anytype) i32 {
    const scalar_type = @TypeOf(value);
    if (value == 0) return -1;
    const U = std.meta.Int(.unsigned, @typeInfo(scalar_type).int.bits);
    return @as(i32, @ctz(@as(U, @bitCast(value))));
}

/// Index of the most significant set bit (0-based), or `-1` when `value == 0`.
/// For signed inputs the two's-complement bit pattern is used, so
/// `findMSB(-1) == 31`. Use `findMSB(value)` as a cheap `floor(log2(value))` for
/// positive integers.
pub fn findMSB(value: anytype) i32 {
    const scalar_type = @TypeOf(value);
    const bits: comptime_int = @typeInfo(scalar_type).int.bits;
    if (value == 0) return -1;
    var v = value;
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
    const scalar_type = @TypeOf(value);
    const U = std.meta.Int(.unsigned, @typeInfo(scalar_type).int.bits);
    const uv: U = @bitCast(value);
    const mask: U = (@as(U, 1) << @intCast(bits)) -% 1;
    const extracted = (uv >> @intCast(offset)) & mask;
    return @bitCast(extracted);
}

/// Replace `bits` bits of `base` (starting at `offset`) with the low `bits`
/// bits of `insert`; all other bits of `base` are kept.
/// Use for packing several small fields into one integer.
pub fn bitfieldInsert(base: anytype, insert: anytype, offset: i32, bits: i32) @TypeOf(base, insert) {
    const scalar_type = @TypeOf(base, insert);
    const U = std.meta.Int(.unsigned, @typeInfo(scalar_type).int.bits);
    const ub: U = @bitCast(base);
    const ui: U = @bitCast(insert);
    const mask: U = (@as(U, 1) << @intCast(bits)) -% 1;
    const shifted_mask = mask << @intCast(offset);
    return @bitCast((ub & ~shifted_mask) | ((ui << @intCast(offset)) & shifted_mask));
}

/// Reverse the order of the bits of `value` (LSB becomes MSB). Use for
/// symmetric hashing of small bit patterns.
pub fn bitfieldReverse(value: anytype) @TypeOf(value) {
    return @bitReverse(value);
}

pub const AddCarryResult = extern struct { sum: u32, carry: u32 };

/// Unsigned addition that reports overflow: returns `sum` (wrapping) and
/// `carry == 1` if `left_hand_side + right_hand_side` exceeded the type's range. Use for multi-word
/// (big number) arithmetic chains.
pub fn uaddCarry(left_hand_side: anytype, right_hand_side: anytype) extern struct { sum: @TypeOf(left_hand_side, right_hand_side), carry: @TypeOf(left_hand_side, right_hand_side) } {
    const scalar_type = @TypeOf(left_hand_side, right_hand_side);
    const r = @addWithOverflow(left_hand_side, right_hand_side);
    return .{ .sum = r[0], .carry = if (r[1] != 0) @as(scalar_type, 1) else @as(scalar_type, 0) };
}

/// Unsigned subtraction that reports underflow: returns `diff` (wrapping)
/// and `borrow == 1` if `right_hand_side > left_hand_side`. Use for multi-word arithmetic chains.
pub fn usubBorrow(left_hand_side: anytype, right_hand_side: anytype) extern struct { diff: @TypeOf(left_hand_side, right_hand_side), borrow: @TypeOf(left_hand_side, right_hand_side) } {
    const scalar_type = @TypeOf(left_hand_side, right_hand_side);
    const r = @subWithOverflow(left_hand_side, right_hand_side);
    return .{ .diff = r[0], .borrow = if (r[1] != 0) @as(scalar_type, 1) else @as(scalar_type, 0) };
}

// ---- reductions ----

/// GLM `compAdd`: sum of all components of a vector. For a scalar this is
/// the identity — the function exists for API symmetry with the vector
/// version.
pub fn compAdd(value: anytype) @TypeOf(value) {
    return value;
}

// ---- ext/scalar_common ----

/// Smallest of three values: the result of two nested `min` calls.
pub fn min3(value1: anytype, value2: anytype, value3: anytype) @TypeOf(value1, value2, value3) {
    return min(min(value1, value2), value3);
}

/// Smallest of four values.
pub fn min4(value1: anytype, value2: anytype, value3: anytype, value4: anytype) @TypeOf(value1, value2, value3, value4) {
    return min(min(value1, value2), min(value3, value4));
}

/// Largest of three values.
pub fn max3(value1: anytype, value2: anytype, value3: anytype) @TypeOf(value1, value2, value3) {
    return max(max(value1, value2), value3);
}

/// Largest of four values.
pub fn max4(value1: anytype, value2: anytype, value3: anytype, value4: anytype) @TypeOf(value1, value2, value3, value4) {
    return max(max(value1, value2), max(value3, value4));
}

/// NaN-tolerant minimum (IEEE `fmin`): if one argument is NaN the other is
/// returned, so a single bad value cannot poison the result. Use for
/// clamping in pipelines where NaN may legitimately appear.
pub fn fmin(left_hand_side: anytype, right_hand_side: anytype) @TypeOf(left_hand_side, right_hand_side) {
    if (isNan(left_hand_side)) return right_hand_side;
    if (isNan(right_hand_side)) return left_hand_side;
    return min(left_hand_side, right_hand_side);
}

/// NaN-tolerant minimum of three values.
pub fn fmin3(value1: anytype, value2: anytype, value3: anytype) @TypeOf(value1, value2, value3) {
    if (isNan(value1)) return fmin(value2, value3);
    if (isNan(value2)) return fmin(value1, value3);
    if (isNan(value3)) return min(value1, value2);
    return min3(value1, value2, value3);
}

/// NaN-tolerant minimum of four values.
pub fn fmin4(value1: anytype, value2: anytype, value3: anytype, value4: anytype) @TypeOf(value1, value2, value3, value4) {
    if (isNan(value1)) return fmin3(value2, value3, value4);
    if (isNan(value2)) return min(value1, fmin(value3, value4));
    if (isNan(value3)) return fmin(min(value1, value2), value4);
    if (isNan(value4)) return min3(value1, value2, value3);
    return min4(value1, value2, value3, value4);
}

/// NaN-tolerant maximum (IEEE `fmax`): NaN arguments are ignored.
pub fn fmax(left_hand_side: anytype, right_hand_side: anytype) @TypeOf(left_hand_side, right_hand_side) {
    if (isNan(left_hand_side)) return right_hand_side;
    if (isNan(right_hand_side)) return left_hand_side;
    return max(left_hand_side, right_hand_side);
}

/// NaN-tolerant maximum of three values.
pub fn fmax3(value1: anytype, value2: anytype, value3: anytype) @TypeOf(value1, value2, value3) {
    if (isNan(value1)) return fmax(value2, value3);
    if (isNan(value2)) return fmax(value1, value3);
    if (isNan(value3)) return max(value1, value2);
    return max3(value1, value2, value3);
}

/// NaN-tolerant maximum of four values.
pub fn fmax4(value1: anytype, value2: anytype, value3: anytype, value4: anytype) @TypeOf(value1, value2, value3, value4) {
    if (isNan(value1)) return fmax3(value2, value3, value4);
    if (isNan(value2)) return max(value1, fmax(value3, value4));
    if (isNan(value3)) return fmax(max(value1, value2), value4);
    if (isNan(value4)) return max3(value1, value2, value3);
    return max4(value1, value2, value3, value4);
}

/// NaN-tolerant clamp: `fmin(fmax(value, min_val), max_val)`. Unlike `clamp` a NaN input
/// does not leak into the result.
pub fn fclamp(value: anytype, min_val: anytype, max_val: anytype) @TypeOf(value, min_val, max_val) {
    return fmin(fmax(value, min_val), max_val);
}

/// Clamp into `[0, 1]`. The standard way to turn any value into a valid
/// interpolation factor, alpha or normalized coordinate.
pub fn clamp01(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    return clamp(value, @as(scalar_type, 0), @as(scalar_type, 1));
}

/// Sawtooth wave in `[0, 1)` with period 1 (same as `fract`). Use to loop
/// a time/coordinate signal: `repeat(t)` gives an endlessly repeating
/// 0→1 ramp.
pub fn repeat(value: anytype) @TypeOf(value) {
    return fract(value);
}

/// Sawtooth of the absolute value: `mirrorClamp(-0.3) == 0.3`. The result
/// stays in `[0, 1)` for any input; use on signed signals that must remain
/// non-negative.
pub fn mirrorClamp(value: anytype) @TypeOf(value) {
    return fract(abs(value));
}

/// Triangle wave in `[0, 1]`: for `value` growing, the output cycles
/// `0 → 1 → 0 → 1 ...`. Use for ping-pong animation loops and mirrored
/// texture tiling.
pub fn mirrorRepeat(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    const abs_x = abs(value);
    const clamp_v = mod(floor(abs_x), @as(scalar_type, 2));
    const floor_v = floor(abs_x);
    const rest = abs_x - floor_v;
    const mirror = clamp_v + rest;
    return mix(rest, @as(scalar_type, 1) - rest, mirror >= @as(scalar_type, 1));
}

/// Round to nearest `i32` using `int(value + 0.5)` (truncating). Note this is
/// asymmetric for negatives (`iround(-1.5) == -1`), matching GLSL `iround`.
/// Use when the result must be fed into an integer API.
pub fn iround(value: anytype) i32 {
    const scalar_type = rtType(@TypeOf(value));
    return @as(i32, @intFromFloat(@as(scalar_type, value) + @as(scalar_type, 0.5)));
}

/// Round to nearest `u32` via `uint(value + 0.5)`; the input is expected to be
/// non-negative.
pub fn uround(value: anytype) u32 {
    const scalar_type = rtType(@TypeOf(value));
    return @as(u32, @intFromFloat(@as(scalar_type, value) + @as(scalar_type, 0.5)));
}

// ---- ext/scalar_reciprocal ----

/// Secant `1/cos(x)`. Singular at `pi/2 + k*pi`.
pub fn sec(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return @as(scalar_type, 1) / cos(@as(scalar_type, value));
}

/// Cosecant `1/sin(x)`. Singular at multiples of `pi`.
pub fn csc(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return @as(scalar_type, 1) / sin(@as(scalar_type, value));
}

/// Cotangent `cos(x)/sin(x)`, computed as `tan(pi/2 - x)` for numerical
/// stability around small angles. Singular at multiples of `pi`.
pub fn cot(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    const pi_over_2: scalar_type = cast(scalar_type, 3.1415926535897932384626433832795 / 2.0);
    return tan(pi_over_2 - @as(scalar_type, value));
}

/// Inverse secant `acos(1/x)`, defined for `|value| >= 1`.
pub fn asec(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return acos(@as(scalar_type, 1) / @as(scalar_type, value));
}

/// Inverse cosecant `asin(1/x)`, defined for `|value| >= 1`.
pub fn acsc(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return asin(@as(scalar_type, 1) / @as(scalar_type, value));
}

/// Inverse cotangent `pi/2 - atan(x)`, result in `(0, pi)`.
pub fn acot(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    const pi_over_2: scalar_type = cast(scalar_type, 3.1415926535897932384626433832795 / 2.0);
    return pi_over_2 - atan(@as(scalar_type, value));
}

/// Hyperbolic secant `1/cosh(x)` — a bell-shaped curve that decays to
/// zero at large |value|.
pub fn sech(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return @as(scalar_type, 1) / cosh(@as(scalar_type, value));
}

/// Hyperbolic cosecant `1/sinh(x)`. Singular at `value == 0`.
pub fn csch(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return @as(scalar_type, 1) / sinh(@as(scalar_type, value));
}

/// Hyperbolic cotangent `cosh(x)/sinh(x)`. Singular at `value == 0`, tends to
/// ±1 at infinity.
pub fn coth(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return cosh(@as(scalar_type, value)) / sinh(@as(scalar_type, value));
}

/// Inverse hyperbolic secant `acosh(1/x)`, defined for `value` in `(0, 1]`.
pub fn asech(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return acosh(@as(scalar_type, 1) / @as(scalar_type, value));
}

/// Inverse hyperbolic cosecant `asinh(1/x)`, defined for `value != 0`.
pub fn acsch(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return asinh(@as(scalar_type, 1) / @as(scalar_type, value));
}

/// Inverse hyperbolic cotangent `atanh(1/x)`, defined for `|value| > 1`.
pub fn acoth(value: anytype) rtType(@TypeOf(value)) {
    const scalar_type = rtType(@TypeOf(value));
    return atanh(@as(scalar_type, 1) / @as(scalar_type, value));
}

// ---- ext/scalar_relational + gtc/epsilon ----

/// Tolerance comparison: true if `|left_hand_side - right_hand_side| <= epsilon`. Use instead of `==`
/// whenever `left_hand_side`/`right_hand_side` are results of floating-point arithmetic.
pub fn equalEps(left_hand_side: anytype, right_hand_side: anytype, epsilon: anytype) bool {
    return abs(left_hand_side - right_hand_side) <= epsilon;
}

/// Complement of `equalEps`: true if `|left_hand_side - right_hand_side| > epsilon`.
pub fn notEqualEps(left_hand_side: anytype, right_hand_side: anytype, epsilon: anytype) bool {
    return abs(left_hand_side - right_hand_side) > epsilon;
}

/// Strict tolerance comparison: true if `|x - y| < eps`. The epsilon
/// variant of GLM's gtc/epsilon module — useful for "is this inside a
/// tolerance ball" checks.
pub fn epsilonEqual(left_hand_side: anytype, right_hand_side: anytype, epsilon: anytype) bool {
    return abs(left_hand_side - right_hand_side) < epsilon;
}

/// Complement of `epsilonEqual`: true if `|x - y| >= eps`.
pub fn epsilonNotEqual(left_hand_side: anytype, right_hand_side: anytype, epsilon: anytype) bool {
    return abs(left_hand_side - right_hand_side) >= epsilon;
}

const FltInt = extern struct {
    fn asI32(value: f32) i32 {
        return @as(i32, @bitCast(value));
    }
    fn asI64(value: f64) i64 {
        return @as(i64, @bitCast(value));
    }
};

/// ULP-based float comparison: true if `left_hand_side` and `right_hand_side` are at most `max_ulps`
/// representable steps apart (bit patterns compared as signed integers,
/// with the sign bit handled separately). Unlike an epsilon test this works
/// at *any* magnitude — `1.0` vs the float just below it is 1 ULP, the same
/// as `1e30` vs its neighbor. Caveat: `+0.0` and `-0.0` compare unequal.
pub fn equalULP(left_hand_side: anytype, right_hand_side: anytype, max_ulps: anytype) bool {
    const scalar_type = @TypeOf(left_hand_side);
    if (scalar_type == f64) {
        const a: i64 = FltInt.asI64(left_hand_side);
        const b: i64 = FltInt.asI64(right_hand_side);
        if ((a < 0) != (b < 0)) return false;
        return abs(a - b) <= @as(i64, max_ulps);
    } else {
        const a: i32 = FltInt.asI32(left_hand_side);
        const b: i32 = FltInt.asI32(right_hand_side);
        if ((a < 0) != (b < 0)) return false;
        return abs(a - b) <= @as(i32, max_ulps);
    }
}

/// Complement of `equalULP`.
pub fn notEqualULP(left_hand_side: anytype, right_hand_side: anytype, max_ulps: anytype) bool {
    return !equalULP(left_hand_side, right_hand_side, max_ulps);
}

// ---- ext/scalar_ulp ----

/// Next representable float strictly greater than `value` (walks toward +inf).
/// Use to iterate a float lattice or to compute "1 ULP above" a value.
pub fn nextFloat(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    return std.math.nextAfter(value, std.math.inf(scalar_type));
}

/// Apply `nextFloat` `ulps` times — moves `ulps` steps up the float
/// lattice in one call.
pub fn nextFloatN(value: anytype, ulps: i32) @TypeOf(value) {
    var temp = value;
    var i: i32 = 0;
    while (i < ulps) : (i += 1) temp = nextFloat(temp);
    return temp;
}

/// Next representable float strictly smaller than `value` (walks toward -inf).
/// Use to compute "1 ULP below" a value.
pub fn prevFloat(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    return std.math.nextAfter(value, -std.math.inf(scalar_type));
}

/// Apply `prevFloat` `ulps` times — moves `ulps` steps down the float
/// lattice in one call.
pub fn prevFloatN(value: anytype, ulps: i32) @TypeOf(value) {
    var temp = value;
    var i: i32 = 0;
    while (i < ulps) : (i += 1) temp = prevFloat(temp);
    return temp;
}

/// Absolute distance between `left_hand_side` and `right_hand_side` measured in ULPs (number of
/// representable floats between them). Note: this variant intentionally
/// skips the sign-bit check of `equalULP`, so `floatDistance(-1, 1)` is a
/// huge number. Use for a magnitude-independent error metric.
pub fn floatDistance(left_hand_side: anytype, right_hand_side: anytype) i64 {
    const scalar_type = @TypeOf(left_hand_side, right_hand_side);
    if (scalar_type == f64) {
        return abs(FltInt.asI64(left_hand_side) - FltInt.asI64(right_hand_side));
    } else {
        const a: f32 = @floatCast(left_hand_side);
        const b: f32 = @floatCast(right_hand_side);
        return abs(@as(i64, FltInt.asI32(a)) - @as(i64, FltInt.asI32(b)));
    }
}

// ---- bit-casts (func_common) ----

/// Reinterpret the bits of an `f32` as `i32` (no conversion).
/// Use to sort/compare floats by bit pattern, or to store data in the
/// mantissa bits. Inverse: `intBitsToFloat`.
pub fn floatBitsToInt(float_bits: f32) i32 {
    return @as(i32, @bitCast(float_bits));
}

/// Reinterpret the bits of an `f32` as `u32`. Same idea as
/// `floatBitsToInt` with an unsigned view.
pub fn floatBitsToUint(float_bits: f32) u32 {
    return @as(u32, @bitCast(float_bits));
}

/// Reconstruct an `f32` from an `i32` bit pattern; the inverse of
/// `floatBitsToInt`.
pub fn intBitsToFloat(int_bits: i32) f32 {
    return @as(f32, @bitCast(int_bits));
}

/// Reconstruct an `f32` from a `u32` bit pattern; the inverse of
/// `floatBitsToUint`.
pub fn uintBitsToFloat(uint_bits: u32) f32 {
    return @as(f32, @bitCast(uint_bits));
}

// ---- ext/scalar_integer + gtc/round ----

/// True if `value` is a power of two (`1, 2, 4, ...`); the sign is ignored
/// for signed types, and `0` also returns true (bit trick, matches GLM).
/// Use to validate sizes before bit-shift scaling.
pub fn isPowerOfTwo(value: anytype) bool {
    const scalar_type = @TypeOf(value);
    const result = if (isSigned(scalar_type)) abs(value) else value;
    return (result & (result - 1)) == 0;
}

/// Truncated integer remainder (C++ `%` semantics: sign follows the
/// dividend). Zig's `%` already behaves this way for ints; the function is
/// a named alias used for clarity inside the round/multiple helpers.
pub fn remInt(dividend: anytype, divisor: anytype) @TypeOf(dividend, divisor) {
    return dividend % divisor;
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
    const scalar_type = @TypeOf(value);
    const bits: comptime_int = @typeInfo(scalar_type).int.bits;
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
    const scalar_type = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    return @as(scalar_type, 1) << @intCast(findMSB(value));
}

/// Nearest power of two; ties resolve upward (`roundPowerOfTwo(5) == 4`,
/// `roundPowerOfTwo(7) == 8`, `roundPowerOfTwo(6) == 8`). Use when you need
/// a power-of-two that "best" approximates a size.
pub fn roundPowerOfTwo(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    const prev = @as(scalar_type, 1) << @intCast(findMSB(value));
    const next = prev << 1;
    return if ((next - value) < (value - prev)) next else prev;
}

/// Alias for `ceilPowerOfTwo`.
pub fn nextPowerOfTwo(value: anytype) @TypeOf(value) {
    return ceilPowerOfTwo(value);
}

/// Alias for `floorPowerOfTwo`.
pub fn prevPowerOfTwo(value: anytype) @TypeOf(value) {
    const scalar_type = @TypeOf(value);
    if (isPowerOfTwo(value)) return value;
    return @as(scalar_type, 1) << @intCast(findMSB(value));
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
    const scalar_type = @TypeOf(source, multiple);
    if (comptime isFloat(scalar_type)) {
        if (source > @as(scalar_type, 0)) {
            return source + (multiple - @rem(source, multiple));
        } else {
            return source + @rem(-source, multiple);
        }
    } else if (comptime isSigned(scalar_type)) {
        if (source > @as(scalar_type, 0)) {
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
    const scalar_type = @TypeOf(source, multiple);
    if (comptime isFloat(scalar_type)) {
        if (source >= @as(scalar_type, 0)) {
            return source - @rem(source, multiple);
        } else {
            return source - @rem(source, multiple) - multiple;
        }
    } else if (source >= @as(scalar_type, 0)) {
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
    const scalar_type = @TypeOf(source, multiple);
    if (comptime isFloat(scalar_type)) {
        if (source >= @as(scalar_type, 0)) {
            return source - @rem(source, multiple);
        } else {
            const tmp = source + @as(scalar_type, 1);
            return tmp - @rem(tmp, multiple) - multiple;
        }
    } else if (source >= @as(scalar_type, 0)) {
        return source - remInt(source, multiple);
    } else {
        const tmp = source + 1;
        return tmp - remInt(tmp, multiple) - multiple;
    }
}

/// Index of the start of the most significant run of `significant_bit_count`
/// consecutive set bits, or `-1` if `value` has fewer set bits in total.
/// Use for locating the topmost dense region of a bitmask.
pub fn findNSB(value: anytype, significant_bit_count: i32) i32 {
    const scalar_type = @TypeOf(value);
    const bits: comptime_int = @typeInfo(scalar_type).int.bits;
    if (bitCount(value) < significant_bit_count) return -1;
    const one: scalar_type = 1;
    var bit_pos: i32 = 0;
    var key = value;
    var n_bit_count = significant_bit_count;
    var bit_step: i32 = @intCast(bits / 2);
    while (key > one) {
        const mask = (@as(scalar_type, 1) << @intCast(bit_step)) - 1;
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
