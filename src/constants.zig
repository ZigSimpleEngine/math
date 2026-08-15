//! GLM constants: ext/scalar_constants + gtc/constants.
//! Values are bit-exact copies of the GLM 1.1.0 literal strings.

const std = @import("std");

pub inline fn cast(comptime T: type, comptime literal: comptime_float) T {
    return @as(T, literal);
}

// ---- ext/scalar_constants ----

pub fn epsilon(comptime T: type) T {
    return if (T == f64) std.math.floatEps(f64) else std.math.floatEps(f32);
}

pub fn pi(comptime T: type) T {
    return cast(T, 3.14159265358979323846264338327950288);
}

pub fn cos_one_over_two(comptime T: type) T {
    return cast(T, 0.877582561890372716130286068203503191);
}

// ---- gtc/constants ----

pub fn zero(comptime T: type) T {
    return cast(T, 0);
}

pub fn one(comptime T: type) T {
    return cast(T, 1);
}

pub fn two_pi(comptime T: type) T {
    return cast(T, 6.28318530717958647692528676655900576);
}

pub fn tau(comptime T: type) T {
    return two_pi(T);
}

pub fn root_pi(comptime T: type) T {
    return cast(T, 1.772453850905516027);
}

pub fn half_pi(comptime T: type) T {
    return cast(T, 1.57079632679489661923132169163975144);
}

pub fn three_over_two_pi(comptime T: type) T {
    return cast(T, 4.71238898038468985769396507491925432);
}

pub fn quarter_pi(comptime T: type) T {
    return cast(T, 0.785398163397448309615660845819875721);
}

pub fn one_over_pi(comptime T: type) T {
    return cast(T, 0.318309886183790671537767526745028724);
}

pub fn one_over_two_pi(comptime T: type) T {
    return cast(T, 0.159154943091895335768883763372514362);
}

pub fn two_over_pi(comptime T: type) T {
    return cast(T, 0.636619772367581343075535053490057448);
}

pub fn four_over_pi(comptime T: type) T {
    return cast(T, 1.273239544735162686151070106980114898);
}

pub fn two_over_root_pi(comptime T: type) T {
    return cast(T, 1.12837916709551257389615890312154517);
}

pub fn one_over_root_two(comptime T: type) T {
    return cast(T, 0.707106781186547524400844362104849039);
}

pub fn root_half_pi(comptime T: type) T {
    return cast(T, 1.253314137315500251);
}

pub fn root_two_pi(comptime T: type) T {
    return cast(T, 2.506628274631000502);
}

pub fn root_ln_four(comptime T: type) T {
    return cast(T, 1.17741002251547469);
}

pub fn e(comptime T: type) T {
    return cast(T, 2.71828182845904523536);
}

pub fn euler(comptime T: type) T {
    return cast(T, 0.577215664901532860606);
}

pub fn root_two(comptime T: type) T {
    return cast(T, 1.41421356237309504880168872420969808);
}

pub fn root_three(comptime T: type) T {
    return cast(T, 1.73205080756887729352744634150587236);
}

pub fn root_five(comptime T: type) T {
    return cast(T, 2.23606797749978969640917366873127623);
}

pub fn ln_two(comptime T: type) T {
    return cast(T, 0.693147180559945309417232121458176568);
}

pub fn ln_ten(comptime T: type) T {
    return cast(T, 2.30258509299404568401799145468436421);
}

pub fn ln_ln_two(comptime T: type) T {
    return cast(T, -0.3665129205816643);
}

pub fn third(comptime T: type) T {
    return cast(T, 1.0 / 3.0);
}

pub fn two_thirds(comptime T: type) T {
    return cast(T, 2.0 / 3.0);
}

pub fn golden_ratio(comptime T: type) T {
    return cast(T, 1.61803398874989484820458683436563811);
}