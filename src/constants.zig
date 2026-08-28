//! Mathematical constants — GLM ext/scalar_constants + gtc/constants.
//!
//! Each function returns the constant in the requested type:
//! `constants.pi(f32)` → `3.14159274` (f32 bit-exact),
//! `constants.pi(f64)` → full 53-bit double precision.
//!
//! The literal values are bit-exact copies of the decimal strings used by
//! GLM 1.1.0 (verified against the GLM reference output). For writing
//! portable code, call the desired constant by name here instead of
//! typing literals — it stays correct in both f32 and f64 builds.

const std = @import("std");

/// Compile-time cast of a decimal literal into `scalar_type`. Using `@as` (not a
/// float-to-float `@floatCast`) preserves the trailing digits GLM's
/// decimal constants carry: the value rounds to the nearest
/// representable float of the target type.
pub inline fn cast(comptime scalar_type: type, comptime literal: comptime_float) scalar_type {
    return @as(scalar_type, literal);
}

// ---- ext/scalar_constants ----

/// Machine epsilon of `scalar_type` (GLM `epsilon<scalar_type>()`): the smallest ε such that
/// `1 + ε != 1`. Use it as the default tolerance in equality helpers:
/// `equalEps(a, b, constants.epsilon(scalar_type))`.
pub fn epsilon(comptime scalar_type: type) scalar_type {
    return if (scalar_type == f64) std.math.floatEps(f64) else std.math.floatEps(f32);
}

/// π — ratio of a circle's circumference to its diameter. The f32 form
/// rounds to `3.14159274` (the standard `float` π).
pub fn pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 3.14159265358979323846264338327950288);
}

/// cos(1/2) ≈ 0.8776 (GLM `cos_one_over_two<scalar_type>()`): the cosine of half a
/// radian — the |w|/|q| threshold quaternion `pow` uses to pick its
/// numerically stable branch. Exposed for parity with GLM.
pub fn cos_one_over_two(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.877582561890372716130286068203503191);
}

// ---- gtc/constants ----

/// Additive identity of `scalar_type`: `x + zero(scalar_type) == x`. Exists for code that
/// builds constants generically over scalar types.
pub fn zero(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0);
}

/// Multiplicative identity of `scalar_type`: `x * one(scalar_type) == x`.
pub fn one(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1);
}

/// 2π — one full revolution in radians. Use for angle wrapping:
/// `angle % two_pi(scalar_type)`.
pub fn two_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 6.28318530717958647692528676655900576);
}

/// τ — alias of `two_pi` (GLM `tau<scalar_type>()`): the modern name for the
/// circle constant; provided for GPU-shader parity.
pub fn tau(comptime scalar_type: type) scalar_type {
    return two_pi(scalar_type);
}

/// √π (GLM `root_pi<scalar_type>()`): the normalization constant of the Gaussian
/// and of the error function.
pub fn root_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.772453850905516027);
}

/// π/2 — a quarter circle in radians.
pub fn half_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.57079632679489661923132169163975144);
}

/// 3π/2 (GLM `three_over_two_pi<scalar_type>()`).
pub fn three_over_two_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 4.71238898038468985769396507491925432);
}

/// π/4 — 45°. Handy for octagon/pre-scaled diagonal offsets: the
/// diagonal neighbor vector of a unit square is `(c, c)` with
/// `c = cos(quarter_pi(scalar_type))`.
pub fn quarter_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.785398163397448309615660845819875721);
}

/// π/180 — degrees→radians conversion factor: `angle_rad = degrees * radians(T)`.
/// The constant behind `scalar.radians`; use it to convert a human-readable
/// angle (e.g. from a UI slider) into something the trig functions expect.
pub fn radians(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.017453292519943295769236907684886127);
}

/// 180/π — radians→degrees conversion factor: `angle_deg = radians_value * degrees(T)`.
/// The constant behind `scalar.degrees`; use it to turn a trig result back
/// into a display angle.
pub fn degrees(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 57.29577951308232087679815481410517);
}

/// 1/π (GLM `one_over_pi<scalar_type>()`).
pub fn one_over_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.318309886183790671537767526745028724);
}

/// 1/(2π) (GLM `one_over_two_pi<scalar_type>()`): multiply an angle by this to get
/// the fraction of the full circle it spans.
pub fn one_over_two_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.159154943091895335768883763372514362);
}

/// 2/π (GLM `two_over_pi<scalar_type>()`).
pub fn two_over_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.636619772367581343075535053490057448);
}

/// 4/π (GLM `four_over_pi<scalar_type>()`): the constant in the fast
/// approximations of atan/sin used by many signal-processing kernels.
pub fn four_over_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.273239544735162686151070106980114898);
}

/// 2/√π (GLM `two_over_root_pi<scalar_type>()`): factor of the Gaussian integral
/// and of `erf`-like S-curves.
pub fn two_over_root_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.12837916709551257389615890312154517);
}

/// 1/√2 ≈ 0.7071 (GLM `one_over_root_two<scalar_type>()`): the cosine of 45° —
/// the unit-balanced diagonal factor when splitting a vector equally
/// into components.
pub fn one_over_root_two(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.707106781186547524400844362104849039);
}

/// √(π/2) (GLM `root_half_pi<scalar_type>()`).
pub fn root_half_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.253314137315500251);
}

/// √(2π) (GLM `root_two_pi<scalar_type>()`): normalization of the standard normal
/// distribution.
pub fn root_two_pi(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 2.506628274631000502);
}

/// √(ln 4) (GLM `root_ln_four<scalar_type>()`).
pub fn root_ln_four(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.17741002251547469);
}

/// Euler's number e — base of natural logarithms.
pub fn e(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 2.71828182845904523536);
}

/// Euler–Mascheroni constant γ ≈ 0.5772 (GLM `euler<scalar_type>()`, also known as
/// gamma): appears in the harmonic series and digamma asymptotics.
pub fn euler(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.577215664901532860606);
}

/// √2 ≈ 1.4142 (GLM `root_two<scalar_type>()`): the diagonal length of a unit
/// square.
pub fn root_two(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.41421356237309504880168872420969808);
}

/// √3 ≈ 1.7321 (GLM `root_three<scalar_type>()`): diagonal of a unit cube and the
/// basis constant of the hexagonal/cubic lattices.
pub fn root_three(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.73205080756887729352744634150587236);
}

/// √5 ≈ 2.2361 (GLM `root_five<scalar_type>()`): appears in pentagon geometry and
/// the golden-ratio identity φ = (1 + √5)/2.
pub fn root_five(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 2.23606797749978969640917366873127623);
}

/// ln 2 (GLM `ln_two<scalar_type>()`): converts octaves/exponents to natural-log
/// domain — `log2(x) * ln_two(scalar_type) == ln(x)`.
pub fn ln_two(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 0.693147180559945309417232121458176568);
}

/// ln 10 (GLM `ln_ten<scalar_type>()`): converts decimal digits to natural-log
/// domain: `log10(x) * ln_ten(scalar_type) == ln(x)`.
pub fn ln_ten(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 2.30258509299404568401799145468436421);
}

/// ln(ln 2) ≈ −0.3665 (GLM `ln_ln_two<scalar_type>()`): a constant of the
/// iterated logarithm asymptotics.
pub fn ln_ln_two(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, -0.3665129205816643);
}

/// 1/3 (GLM `third<scalar_type>()`): exact decimal, so the f32/f64 forms are the
/// same 0.333… as a rounded literal would give.
pub fn third(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.0 / 3.0);
}

/// 2/3 (GLM `two_thirds<scalar_type>()`).
pub fn two_thirds(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 2.0 / 3.0);
}

/// Golden ratio φ ≈ 1.6180 (GLM `golden_ratio<scalar_type>()`): the aesthetically
/// pleasing proportion used in layout and spiral (Fibonacci) shaping.
pub fn golden_ratio(comptime scalar_type: type) scalar_type {
    return cast(scalar_type, 1.61803398874989484820458683436563811);
}