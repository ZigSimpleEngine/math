//! Vector math — a GLM-compatible `vec<L, T>`.
//!
//! `Vec(L, T)` stores `L` components of type `T` in a Zig SIMD `@Vector(L, T)`
//! and exposes per-component and geometric operations as methods, mirroring
//! GLM's free functions with the receiver as first argument
//! (`a.dot(b)` == GLM `dot(a, b)`).
//!
//! Most per-component methods accept either a full vector or a plain scalar
//! as their argument (`v.mul(2)` works like `v.mul(vec2(2, 2))`), matching
//! GLSL's mixed operand rules. Methods that delegate to `scalar.zig`
//! replicate its NaN/rounding/ULP semantics.

const std = @import("std");
const scalar = @import("scalar.zig");

/// Returns `true` if `T` is a vector type produced by `Vec(L, T)` (it has
/// the `len` declaration and a `v` field). Use to dispatch vector vs scalar
/// arguments in generic code.
pub fn isVec(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "len") and @hasField(T, "v");
}

const floatType = scalar.floatType;

/// Create a vector type with `L` components of type `T` (float, int or bool).
/// The type name doubles as a namespace: `vec3.zero()`, `Vec(4, f32).one()`.
pub fn Vec(comptime L: usize, comptime T: type) type {
    return struct {
        pub const Self = @This();
        pub const len: comptime_int = L;
        pub const value_type: type = T;
        pub const storage_type: type = @Vector(L, T);
        pub const float_type: type = floatType(T);

        v: storage_type,

        // ---- constructors ----

        /// Vector with all components set to `0`. The zero vector is the
        /// additive identity: `v.add(vec3.zero()) == v`.
        pub inline fn zero() Self {
            return .{ .v = @splat(scalar.cast(T, 0)) };
        }

        /// Vector with all components set to `1`. With floats this is the
        /// multiplicative identity; with ints it is the bitmap of all-zero
        /// bits.
        pub inline fn one() Self {
            return .{ .v = @splat(scalar.cast(T, 1)) };
        }

        /// Vector with every component set to `v`: `vec3.fill(0.5)`.
        pub inline fn fill(v: anytype) Self {
            return .{ .v = @splat(scalar.cast(T, v)) };
        }

        /// Basis vector with a single `1` at component `i` (0-based):
        /// `vec3.unit(1) == vec3(0, 1, 0)`. Use for axis-aligned
        /// directions.
        pub inline fn unit(comptime i: usize) Self {
            var r: storage_type = @splat(scalar.cast(T, 0));
            r[i] = scalar.cast(T, 1);
            return .{ .v = r };
        }

        /// Build a vector from flexible arguments, GLM-constructor style:
        /// - a scalar: splatted into every component (`vec3.init(5)`),
        /// - another vector: components copied in order (truncation
        ///   happens naturally: `vec4.init(v3)` keeps the first three),
        /// - a tuple of scalars and/or vectors, concatenated in order:
        ///   `vec4.init(.{ v2, 1.0, 2.0 })`, `vec3.init(.{ v2, v2.x() })`.
        /// All values are cast to `T`. Use for building points/colors from
        /// parts of existing vectors.
        pub fn init(args: anytype) Self {
            const AT = @TypeOf(args);
            if (comptime AT == Self) return args;
            if (comptime isVec(AT)) {
                var r: storage_type = undefined;
                inline for (0..L) |i| r[i] = scalar.cast(T, args.v[i]);
                return .{ .v = r };
            }
            if (comptime scalar.isNumber(AT)) return fill(args);
            const fields = @typeInfo(AT).@"struct".fields;
            var r: storage_type = undefined;
            comptime var n: usize = 0;
            inline for (fields) |f| {
                const e = @field(args, f.name);
                const ET = @TypeOf(e);
                if (comptime isVec(ET)) {
                    inline for (0..@TypeOf(e).len) |k| {
                        if (comptime n >= L) @compileError("Vec init: too many components");
                        r[n] = scalar.cast(T, e.v[k]);
                        n += 1;
                    }
                } else {
                    if (comptime n >= L) @compileError("Vec init: too many components");
                    r[n] = scalar.cast(T, e);
                    n += 1;
                }
            }
            if (comptime n != L) @compileError("Vec init: expected " ++ comptimePrint("{d}", .{L}) ++ " components, got " ++ comptimePrint("{d}", .{n}));
            return .{ .v = r };
        }

        // ---- accessors ----

        /// Read component `i` (0-based, runtime value). Prefer `x`/`y`/`z`/`w`
        /// when the index is known at compile time — they compile to direct
        /// vector lane access without bounds checks.
        pub inline fn get(self: Self, i: usize) T {
            const p: *const storage_type = &self.v;
            return p[i];
        }

        /// Write component `i` of a mutable vector: `v.set(2, 3.0)`.
        pub inline fn set(self: *Self, i: usize, val: T) void {
            self.v[i] = val;
        }

        /// First component (x). GLSL `.x` selector.
        pub inline fn x(self: Self) T {
            return self.v[0];
        }

        /// Second component (y). Compile-time error for vectors of length 1.
        pub inline fn y(self: Self) T {
            if (comptime L < 2) @compileError("vector has no y component");
            return self.v[1];
        }

        /// Third component (z). Compile-time error for vectors of length < 3.
        pub inline fn z(self: Self) T {
            if (comptime L < 3) @compileError("vector has no z component");
            return self.v[2];
        }

        /// Fourth component (w). Compile-time error for vectors of length < 4.
        pub inline fn w(self: Self) T {
            if (comptime L < 4) @compileError("vector has no w component");
            return self.v[3];
        }

        /// Set the first component of a mutable vector.
        pub inline fn setX(self: *Self, val: T) void {
            self.v[0] = val;
        }

        /// Set the second component of a mutable vector.
        pub inline fn setY(self: *Self, val: T) void {
            if (comptime L < 2) @compileError("vector has no y component");
            self.v[1] = val;
        }

        /// Set the third component of a mutable vector.
        pub inline fn setZ(self: *Self, val: T) void {
            if (comptime L < 3) @compileError("vector has no z component");
            self.v[2] = val;
        }

        /// Set the fourth component of a mutable vector.
        pub inline fn setW(self: *Self, val: T) void {
            if (comptime L < 4) @compileError("vector has no w component");
            self.v[3] = val;
        }

        // ---- swizzles ----

        /// Build a 2-component vector from components `a` and `b`. This is
        /// the GLSL swizzle mechanism: `v.zy()` reorders lanes; the source
        /// vector is copied, never modified.
        inline fn swz2(self: Self, comptime a: usize, comptime b: usize) Vec(2, T) {
            return .{ .v = @Vector(2, T){ self.v[a], self.v[b] } };
        }

        /// Build a 3-component vector from any three components of `self`
        /// (GLSL swizzle, e.g. `v.zxy()`). Copies, never reorders in place.
        inline fn swz3(self: Self, comptime a: usize, comptime b: usize, comptime c: usize) Vec(3, T) {
            return .{ .v = @Vector(3, T){ self.v[a], self.v[b], self.v[c] } };
        }

        /// Components (x, y) as a 2-vector: the standard way to drop z/w.
        pub inline fn xy(self: Self) Vec(2, T) {
            if (comptime L < 2) @compileError("swizzle out of range");
            return self.swz2(0, 1);
        }

        /// Components (x, z) as a 2-vector. Typical use: extract the ground
        /// plane position of a 3D point, dropping height.
        pub inline fn xz(self: Self) Vec(2, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz2(0, 2);
        }

        /// Components (y, z) as a 2-vector.
        pub inline fn yz(self: Self) Vec(2, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz2(1, 2);
        }

        /// Components (x, w) as a 2-vector.
        pub inline fn xw(self: Self) Vec(2, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return self.swz2(0, 3);
        }

        /// Components (y, w) as a 2-vector.
        pub inline fn yw(self: Self) Vec(2, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return self.swz2(1, 3);
        }

        /// Components (z, w) as a 2-vector.
        pub inline fn zw(self: Self) Vec(2, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return self.swz2(2, 3);
        }

        /// Components (x, y, z) as a 3-vector: the usual way to get the
        /// positional part of a homogeneous 4-vector.
        pub inline fn xyz(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(0, 1, 2);
        }

        /// Components (x, z, y) as a 3-vector.
        pub inline fn xzy(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(0, 2, 1);
        }

        /// Components (y, x, z) as a 3-vector.
        pub inline fn yxz(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(1, 0, 2);
        }

        /// Components (y, z, x) as a 3-vector.
        pub inline fn yzx(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(1, 2, 0);
        }

        /// Components (z, x, y) as a 3-vector.
        pub inline fn zxy(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(2, 0, 1);
        }

        /// Components (z, y, x) as a 3-vector.
        pub inline fn zyx(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(2, 1, 0);
        }

        /// All four components as a 4-vector. On a vec4 this is a copy; on a
        /// vec2/vec3 it pads with zeroes.
        pub inline fn xyzw(self: Self) Vec(4, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return .{ .v = @Vector(4, T){ self.v[0], self.v[1], self.v[2], self.v[3] } };
        }

        // ---- element-wise application helpers ----

        /// Apply unary scalar function `f` to every component:
        /// `v.apply(scalar.sin)` == `v.sin()`. Backend used by most
        /// per-component methods.
        fn apply(self: Self, comptime f: anytype) Self {
            var r: storage_type = undefined;
            inline for (0..L) |i| r[i] = f(self.v[i]);
            return .{ .v = r };
        }

        /// Apply binary scalar function `f(self[i], b)` per component; `b`
        /// may be a full vector (component-wise) or a scalar (broadcast to
        /// every lane). Backend of min/max/pow/fmin, etc.
        fn apply2(self: Self, b: anytype, comptime f: anytype) Self {
            const BT = @TypeOf(b);
            if (comptime isVec(BT)) {
                var r: storage_type = undefined;
                inline for (0..L) |i| r[i] = f(self.v[i], b.v[i]);
                return .{ .v = r };
            } else {
                var r: storage_type = undefined;
                inline for (0..L) |i| r[i] = f(self.v[i], b);
                return .{ .v = r };
            }
        }

        /// Apply ternary scalar function `f(b, c, self[i])` per component
        /// with mixed vector/scalar operands; the receiver is passed last so
        /// that `self.clamp(lo, hi)` == GLM `clamp(self, lo, hi)`. Backend of
        /// clamp/smoothstep/fclamp.
        fn apply3(self: Self, b: anytype, c: anytype, comptime f: anytype) Self {
            const BT = @TypeOf(b);
            const CT = @TypeOf(c);
            const bv = comptime isVec(BT);
            const cv = comptime isVec(CT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                const bi = if (bv) b.v[i] else b;
                const ci = if (cv) c.v[i] else c;
                r[i] = f(bi, ci, self.v[i]);
            }
            return .{ .v = r };
        }

        // ---- arithmetic ----

        /// Component-wise addition. `rhs` may be a vector (lane-wise) or a
        /// scalar (added to every lane): `pos.add(vel.mul(dt))`,
        /// `v.add(1)`.
        pub inline fn add(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v + rhs.v };
            return .{ .v = self.v + @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        /// Component-wise subtraction. `rhs` may be a vector or a scalar:
        /// `a.sub(b)` == GLSL `a - b`. Use for displacement between points.
        pub inline fn sub(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v - rhs.v };
            return .{ .v = self.v - @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        /// Component-wise multiplication. `rhs` may be a vector (Hadamard
        /// product, NOT the dot product — use `dot`) or a scalar: scaling a
        /// direction by speed is `dir.mul(10)`.
        pub inline fn mul(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v * rhs.v };
            return .{ .v = self.v * @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        /// Component-wise division. `rhs` may be a vector or a scalar.
        pub inline fn div(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v / rhs.v };
            return .{ .v = self.v / @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        /// Negate every component: `v.neg()` == GLSL `-v`.
        pub inline fn neg(self: Self) Self {
            return .{ .v = -self.v };
        }

        /// Component-wise modulo (GLSL `mod(x, y)`, floored, sign follows
        /// the divisor). `rhs` may be a vector or a scalar: wrap an angle in
        /// [0, 2π) with `.mod(2pi)`.
        pub inline fn mod(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.mod);
        }

        /// Component-wise reciprocal (GLM `inverse(vec)`): `1/x` per lane,
        /// computed via a single SIMD division. Use on a `1/d` precomputed
        /// vector, not on matrices (that is `mat.inverse()`).
        pub inline fn inverse(self: Self) Self {
            return .{ .v = @as(storage_type, @splat(scalar.cast(T, 1))) / self.v };
        }

        // ---- relational ----

        /// Backend of the six comparison methods: applies `op` between
        /// `self`'s lanes and `rhs` lanes (scalar rhs is broadcast), yielding
        /// a bool vector whose lanes are true where the comparison holds.
        fn cmp(self: Self, rhs: anytype, comptime op: enum { lt, le, gt, ge, eq, ne }) Vec(L, bool) {
            const sv = @Vector(L, T);
            const b: sv = if (comptime isVec(@TypeOf(rhs))) rhs.v else @splat(scalar.cast(T, rhs));
            return switch (op) {
                .lt => .{ .v = self.v < b },
                .le => .{ .v = self.v <= b },
                .gt => .{ .v = self.v > b },
                .ge => .{ .v = self.v >= b },
                .eq => .{ .v = self.v == b },
                .ne => .{ .v = self.v != b },
            };
        }

        /// Component-wise `<` (GLM `lessThan`). Result lane i is true iff
        /// `self[i] < rhs[i]`; scalar rhs is compared against every lane.
        pub inline fn lessThan(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .lt);
        }

        /// Component-wise `<=` (GLM `lessThanEqual`).
        pub inline fn lessThanEqual(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .le);
        }

        /// Component-wise `>` (GLM `greaterThan`).
        pub inline fn greaterThan(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .gt);
        }

        /// Component-wise `>=` (GLM `greaterThanEqual`).
        pub inline fn greaterThanEqual(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .ge);
        }

        /// Component-wise `==` (GLM `equal`). Exact bitwise equality on ints;
        /// on floats use `equalEps`/`equalULP` for tolerance comparisons.
        pub inline fn equal(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .eq);
        }

        /// Component-wise `!=` (GLM `notEqual`).
        pub inline fn notEqual(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .ne);
        }

        // ---- bool-vector reductions ----

        /// Logical OR over a bool vector's lanes (GLSL `any`): true iff at
        /// least one lane is true, e.g. `v.lessThan(other).any()`. Compile
        /// error unless the vector's component type is `bool`.
        pub inline fn any(self: Self) bool {
            if (comptime T != bool) @compileError("any() requires a bool vector");
            return @reduce(.Or, self.v);
        }

        /// Logical AND over a bool vector's lanes (GLSL `all`): true iff
        /// every lane is true, e.g. `v.greaterThan(bounds).all()` as a
        /// bounds check. Compile error unless the component type is `bool`.
        pub inline fn all(self: Self) bool {
            if (comptime T != bool) @compileError("all() requires a bool vector");
            return @reduce(.And, self.v);
        }

        // ---- common ----

        /// Component-wise absolute value (GLM `abs`). NaN propagates; on
        /// integer vectors `@abs` is avoided because it misbehaves on
        /// `comptime` parameters — see `scalar.abs`.
        pub inline fn abs(self: Self) Self {
            return self.apply(scalar.abs);
        }

        /// Component-wise sign (GLM `sign`): -1 / 0 / +1 per lane, sign of
        /// 0 is 0 and NaN yields NaN. See `scalar.sign` for the exact rules.
        pub inline fn sign(self: Self) Self {
            return self.apply(scalar.sign);
        }

        /// Component-wise floor (GLM `floor`): largest integer ≤ value,
        /// as a float. E.g. `t.floor().mul(cell_size)` snaps to a grid.
        pub inline fn floor(self: Self) Self {
            return self.apply(scalar.floor);
        }

        /// Component-wise ceil (GLM `ceil`): smallest integer ≥ value.
        pub inline fn ceil(self: Self) Self {
            return self.apply(scalar.ceil);
        }

        /// Component-wise rounding away from zero (GLM `round`): 2.5 → 3,
        /// -2.5 → -3. For banker's rounding use `roundEven`.
        pub inline fn round(self: Self) Self {
            return self.apply(scalar.round);
        }

        /// Component-wise round-half-to-even (GLM `roundEven`): 2.5 → 2,
        /// 3.5 → 4. Use when dividing into pairs/groups to avoid bias.
        pub inline fn roundEven(self: Self) Self {
            return self.apply(scalar.roundEven);
        }

        /// Component-wise truncation toward zero (GLM `trunc`): fractional
        /// part is dropped. `trunc(-1.7) == -1`.
        pub inline fn trunc(self: Self) Self {
            return self.apply(scalar.trunc);
        }

        /// Component-wise fractional part (GLM `fract`): `x - floor(x)`, so
        /// always in [0, 1). Combine with `floor` to separate a position
        /// into tile index and in-tile offset.
        pub inline fn fract(self: Self) Self {
            return self.apply(scalar.fract);
        }

        /// Component-wise minimum (GLM `min`). `rhs` may be a vector or a
        /// scalar: `pos.clamp(0, size)` takes vectors, `v.max(0)` clamps a
        /// side. NaN handling matches `scalar.min`.
        pub inline fn min(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.min);
        }

        /// Component-wise maximum (GLM `max`).
        pub inline fn max(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.max);
        }

        /// Component-wise clamp into [lo, hi] (GLM `clamp`). `lo`/`hi` may
        /// each be vectors or scalars, mixed freely. `mix` order differs
        /// from GLSL: this is `clamp(x, lo, hi)`.
        pub inline fn clamp(self: Self, lo: anytype, hi: anytype) Self {
            return self.apply3(lo, hi, clampHelper);
        }

        fn clampHelper(lo: anytype, hi: anytype, v: anytype) @TypeOf(v) {
            return scalar.clamp(v, lo, hi);
        }

        /// Linear interpolation between the receiver and `rhs` by factor `a`
        /// (GLM `mix(x, y, a)`, GLSL writes it `mix(x, y, a)` too).
        /// - `a` scalar: uniform blend — `a = 0` returns the receiver, `a = 1`
        ///   returns `rhs`, outside [0,1] it extrapolates
        ///   (`a.neg().mul(t).add(a.mul(1 - t))` is the GLSL formula),
        /// - `a` vector: per-component factors,
        /// - `a` bool (or bool vector): selects component-wise with no
        ///   interpolation (`a` true picks `rhs`). Same effect as `if`.
        pub fn mix(self: Self, rhs: anytype, a: anytype) Self {
            const AT = @TypeOf(a);
            if (comptime AT == bool) return if (a) rhs else self;
            const RT = @TypeOf(rhs);
            const av = comptime isVec(AT);
            const rv = comptime isVec(RT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                const ri = if (rv) rhs.v[i] else rhs;
                const ai = if (av) a.v[i] else a;
                r[i] = scalar.mix(self.v[i], ri, ai);
            }
            return .{ .v = r };
        }

        /// Per-component step function (GLM `step(edge, self)`): 0 where the
        /// receiver is below `edge`, 1 otherwise — i.e. `x >= edge ? 1 : 0`.
        /// Classic use: gate a value by a threshold. `edge` may be vector or
        /// scalar, so a single call can step against a vector of thresholds.
        pub fn step(self: Self, edge: anytype) Self {
            const ET = @TypeOf(edge);
            const ev = comptime isVec(ET);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                const ei = if (ev) edge.v[i] else edge;
                r[i] = scalar.step(ei, self.v[i]);
            }
            return .{ .v = r };
        }

        /// Per-component smoothstep (GLM `smoothstep(e0, e1, self)`): 0 at
        /// `e0`, 1 at `e1` with a Hermite (2t³−3t²) ramp in between. Ideal
        /// for easing transitions; where t leaves [e0,e1] the result clamps.
        pub inline fn smoothstep(self: Self, e0: anytype, e1: anytype) Self {
            return self.apply3(e0, e1, scalar.smoothstep);
        }

        /// Component-wise fused multiply-add: `self·rhs + c` with a single
        /// rounding. `rhs`/`c` may be vectors or scalars (both vector → SIMD
        /// `@mulAdd`, mixed → per-lane scalar path).
        pub inline fn fma(self: Self, rhs: anytype, c: anytype) Self {
            const RT = @TypeOf(rhs);
            const CT = @TypeOf(c);
            if (comptime isVec(RT) and isVec(CT))
                return .{ .v = @mulAdd(storage_type, self.v, rhs.v, c.v) };
            return self.apply3(rhs, c, scalar.fma);
        }

        /// Split each component into fractional and integral parts
        /// (GLM `modf`): `modf(2.75).fract == 0.75`,
        /// `modf(2.75).integral == 2.0`. Both parts keep the sign of the
        /// input and their sum equals it.
        pub fn modf(self: Self) struct { fract: Self, integral: Self } {
            var fr: storage_type = undefined;
            var it: storage_type = undefined;
            inline for (0..L) |i| {
                const m = scalar.modf(self.v[i]);
                fr[i] = m.fract;
                it[i] = m.integral;
            }
            return .{ .fract = .{ .v = fr }, .integral = .{ .v = it } };
        }

        /// Split each component into significand and binary exponent
        /// (GLM `frexp`): `x = significand · 2^exponent` with significand in
        /// [0.5, 1). Usually combined with `ldexp` to move a value between
        /// precisions or to serialize it losslessly.
        pub fn frexp(self: Self) struct { significand: Self, exponent: Vec(L, i32) } {
            var sg: storage_type = undefined;
            var ex: @Vector(L, i32) = undefined;
            inline for (0..L) |i| {
                const m = scalar.frexp(self.v[i]);
                sg[i] = m.significand;
                ex[i] = m.exponent;
            }
            return .{ .significand = .{ .v = sg }, .exponent = .{ .v = ex } };
        }

        /// Component-wise `x · 2^e` (GLM `ldexp`, GLSL `ldexp(x, exp)`).
        /// `e` may be a vector of ints or a single int. The inverse of
        /// `frexp`: `v.frexp().significand.ldexp(v.frexp().exponent) == v`.
        pub fn ldexp(self: Self, e: anytype) Self {
            const ET = @TypeOf(e);
            var r: storage_type = undefined;
            if (comptime isVec(ET)) {
                inline for (0..L) |i| r[i] = scalar.ldexp(self.v[i], e.v[i]);
            } else {
                inline for (0..L) |i| r[i] = scalar.ldexp(self.v[i], e);
            }
            return .{ .v = r };
        }

        /// Per-component NaN test (GLM `isnan`), resulting in a bool vector.
        /// Useful to validate shader output: `v.sub(prev_v).isNan().any()`
        /// detects NaN infecting an animation variable.
        pub fn isNan(self: Self) Vec(L, bool) {
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = scalar.isNan(self.v[i]);
            return .{ .v = r };
        }

        /// Per-component infinity test (GLM `isinf`), resulting in a bool
        /// vector. Detects division-by-zero results from physics steps or
        /// from invalid rotation angles.
        pub fn isInf(self: Self) Vec(L, bool) {
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = scalar.isInf(self.v[i]);
            return .{ .v = r };
        }

        // ---- trigonometric ----

        /// Convert degrees to radians (GLM `radians`): `x · π/180`.
        /// The inverse of `degrees`. Typical use: turn a user-facing angle
        /// (from a UI slider, e.g. 0–360) into something `sin` expects.
        pub inline fn radians(self: Self) Self {
            return self.apply(scalar.radians);
        }

        /// Convert radians to degrees (GLM `degrees`): `x · 180/π`.
        pub inline fn degrees(self: Self) Self {
            return self.apply(scalar.degrees);
        }

        /// Per-component sine (GLM `sin`); the argument is in radians.
        /// Use to produce periodic motion: `y = center + amp * t.sin()`
        /// oscillates between `center ∓ amp`.
        pub inline fn sin(self: Self) Self {
            return self.apply(scalar.sin);
        }

        /// Per-component cosine (GLM `cos`); argument in radians.
        pub inline fn cos(self: Self) Self {
            return self.apply(scalar.cos);
        }

        /// Per-component tangent (GLM `tan`); argument in radians.
        pub inline fn tan(self: Self) Self {
            return self.apply(scalar.tan);
        }

        /// Per-component arc sine (GLM `asin`): result in [-π/2, π/2].
        /// Undefined outside the domain [-1, 1].
        pub inline fn asin(self: Self) Self {
            return self.apply(scalar.asin);
        }

        /// Per-component arc cosine (GLM `acos`): result in [0, π].
        /// Undefined outside the domain [-1, 1].
        pub inline fn acos(self: Self) Self {
            return self.apply(scalar.acos);
        }

        /// Per-component arc tangent (GLM `atan`): result in [-π/2, π/2].
        pub inline fn atan(self: Self) Self {
            return self.apply(scalar.atan);
        }

        /// Per-component atan2 (GLM `atan(self, rhs)`, i.e. the receiver is
        /// the y-axis argument): angle of the polar coordinates (rhs, self),
        /// result in [-π, π]. Use to get the heading of a 2D direction:
        /// `dir.y().atan2(dir.x())`.
        pub inline fn atan2(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.atan2);
        }

        /// Per-component hyperbolic sine (GLM `sinh`). Use for catenary
        /// curves and smooth monotone growth.
        pub inline fn sinh(self: Self) Self {
            return self.apply(scalar.sinh);
        }

        /// Per-component hyperbolic cosine (GLM `cosh`).
        pub inline fn cosh(self: Self) Self {
            return self.apply(scalar.cosh);
        }

        /// Per-component hyperbolic tangent (GLM `tanh`): saturating
        /// function to (-1, 1), useful as an activation curve.
        pub inline fn tanh(self: Self) Self {
            return self.apply(scalar.tanh);
        }

        /// Per-component inverse hyperbolic sine (GLM `asinh`).
        pub inline fn asinh(self: Self) Self {
            return self.apply(scalar.asinh);
        }

        /// Per-component inverse hyperbolic cosine (GLM `acosh`); domain
        /// [1, ∞).
        pub inline fn acosh(self: Self) Self {
            return self.apply(scalar.acosh);
        }

        /// Per-component inverse hyperbolic tangent (GLM `atanh`); domain
        /// (-1, 1).
        pub inline fn atanh(self: Self) Self {
            return self.apply(scalar.atanh);
        }

        // ---- exponential ----

        /// Component-wise power (GLM `pow`): `self[i]^rhs[i]`; `rhs` may be
        /// a vector or scalar. See `scalar.pow` for edge-case rules
        /// (negative bases with non-integer exponents give NaN).
        pub inline fn pow(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.pow);
        }

        /// Component-wise natural exponent (GLM `exp`): `e^x`.
        /// The inverse of `log`.
        pub inline fn exp(self: Self) Self {
            return self.apply(scalar.exp);
        }

        /// Component-wise base-2 exponent (GLM `exp2`): `2^x`. This is the
        /// natural counterpart of `log2`, and is a cheap way to scale
        /// decibels or octaves into a linear domain.
        pub inline fn exp2(self: Self) Self {
            return self.apply(scalar.exp2);
        }

        /// Component-wise natural logarithm (GLM `log`); undefined for
        /// x ≤ 0 (NaN/±inf, see `scalar.log`).
        pub inline fn log(self: Self) Self {
            return self.apply(scalar.log);
        }

        /// Component-wise base-2 logarithm (GLM `log2`); undefined for
        /// x ≤ 0. Use to convert a linear gain to octaves/EV stops.
        pub inline fn log2(self: Self) Self {
            return self.apply(scalar.log2);
        }

        /// Component-wise square root (GLM `sqrt`); undefined (NaN) for
        /// negative inputs.
        pub inline fn sqrt(self: Self) Self {
            return self.apply(scalar.sqrt);
        }

        /// Component-wise inverse square root (GLM `inversesqrt`):
        /// `1/sqrt(x)`. Cheaper to compute than a division by `sqrt` and
        /// used by vector normalization (see `normalize`). Undefined for
        /// x ≤ 0.
        pub inline fn inversesqrt(self: Self) Self {
            return self.apply(scalar.inversesqrt);
        }

        // ---- geometric ----

        // ---- geometric ----

        /// Dot product (GLM/GLSL `dot`): sum of component products. For unit
        /// vectors the result is the cosine of the angle between them;
        /// `n.dot(d)` with a plane normal answers which side of a plane a
        /// direction is on (see `faceforward`).
        pub inline fn dot(self: Self, b: Self) T {
            return @reduce(.Add, self.v * b.v);
        }

        /// Euclidean length `√(x·x)` (GLM `length`). Computed in the
        /// vector's float type (int vectors promote to `float_type`, so
        /// even `ivec2(3,4).length() == 5.0`). Use `distance` for the
        /// length of a difference.
        pub inline fn length(self: Self) float_type {
            const d = @reduce(.Add, self.v * self.v);
            return scalar.sqrt(scalar.cast(float_type, d));
        }

        /// Distance between two points (GLM `distance`): `(a - b).length()`.
        /// For comparing distances prefer `lengthSquared`-style checks
        /// (`a.sub(b).dot(a.sub(b))`) when the square root can be avoided.
        pub inline fn distance(self: Self, b: Self) float_type {
            return self.sub(b).length();
        }

        /// Cross product, 3D only (GLM `cross`): a vector perpendicular to
        /// both operands whose length equals the parallelogram area. Handed
        /// right-handed, matching GLM; aggregate into `mat4`-style rotation
        /// bases with `normalize`.
        pub fn cross(self: Self, b: Self) Self {
            if (comptime L != 3) @compileError("cross is only defined for 3-component vectors");
            return .{ .v = @Vector(3, T){
                self.v[1] * b.v[2] - self.v[2] * b.v[1],
                self.v[2] * b.v[0] - self.v[0] * b.v[2],
                self.v[0] * b.v[1] - self.v[1] * b.v[0],
            } };
        }

        /// Unit vector in the same direction (GLM `normalize`):
        /// `v / |v|`. Zero vectors stay zero; returns a float vector even
        /// for int inputs. Use on directions before `dot`/`cross` so that
        /// angle computations behave like on unit spheres.
        pub fn normalize(self: Self) Vec(L, float_type) {
            const d = @reduce(.Add, self.v * self.v);
            const is: float_type = scalar.inversesqrt(scalar.cast(float_type, d));
            const c: @Vector(L, float_type) = if (T == float_type)
                self.v
            else
                @floatCast(self.v);
            return .{ .v = c * @as(@Vector(L, float_type), @splat(is)) };
        }

        /// Orient a normal `self` away from a reference direction (GLM
        /// `faceforward(N, I, Nref)`): returns `self` if `I·Nref < 0`,
        /// otherwise `-self`. For example, flip a surface normal so it
        /// points at the camera: `normal.faceforward(view_dir, normal_ref)`.
        pub inline fn faceforward(self: Self, i: Self, nref: Self) Self {
            return if (nref.dot(i) < 0) self else self.neg();
        }

        /// Reflect an incident vector `self` about a normal `n` (GLM
        /// `reflect(I, N)`): `I - 2·(I·N)·N`. `n` should be normalized;
        /// result is the mirror of `self` across the plane with normal `n`.
        pub inline fn reflect(self: Self, n: Self) Self {
            return self.sub(n.mul(self.dot(n)).mul(@as(T, 2)));
        }

        /// Refract an incident vector `self` at a surface with normal `n`
        /// and index ratio `eta` (GLM `refract(I, N, eta)` =
        /// `refract(I, N, IOR₁/IOR₂)`). Returns the zero vector when total
        /// internal reflection occurs (k < 0). Handy for simulating glass
        /// or water rays without a full ray tracer.
        pub fn refract(self: Self, n: Self, eta: anytype) Self {
            const et: T = scalar.cast(T, eta);
            const dot_value = self.dot(n);
            const k = @as(T, 1) - et * et * (@as(T, 1) - dot_value * dot_value);
            if (k < 0) return Self.zero();
            return self.mul(et).sub(n.mul(scalar.sqrt(k) + et * dot_value));
        }

        // ---- integer / bit ----

        /// Per-component popcount (GLM `bitCount`): number of set bits in
        /// each lane. Result is i32 per GLM's `bitCount(vec)` overloads;
        /// handy for parity checks and bit-packed grid neighbors.
        pub inline fn bitCount(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.bitCount(self.v[i]);
            return .{ .v = r };
        }

        /// Per-component index of the least significant set bit (GLM
        /// `findLSB`), or -1 for a zero lane. Use to decompose a bitmask
        /// into indices.
        pub inline fn findLSB(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.findLSB(self.v[i]);
            return .{ .v = r };
        }

        /// Per-component index of the most significant set bit (GLM
        /// `findMSB`), or -1 for zero. On signed values the sign bit is
        /// interpreted as the MSB, so `findMSB` on -1 yields the top bit.
        pub inline fn findMSB(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.findMSB(self.v[i]);
            return .{ .v = r };
        }

        /// Per-component bitfield extract (GLM `bitfieldExtract`): take
        /// `bits` bits starting at `offset` (LSB origin). `offset`/`bits`
        /// may be vectors or scalars; out-of-range bits are zero-filled.
        pub inline fn bitfieldExtract(self: Self, offset: anytype, bits: anytype) Self {
            var r: storage_type = undefined;
            if (comptime isVec(@TypeOf(offset))) {
                inline for (0..L) |i| r[i] = scalar.bitfieldExtract(self.v[i], offset.v[i], bits.v[i]);
            } else {
                inline for (0..L) |i| r[i] = scalar.bitfieldExtract(self.v[i], offset, bits);
            }
            return .{ .v = r };
        }

        /// Per-component bitfield insert (GLM `bitfieldInsert`): overlay the
        /// low `bits` bits of `insert` into `self` at `offset` (LSB origin).
        /// `insert`/`offset`/`bits` may be vectors or scalars. Inverse of
        /// `bitfieldExtract` for the same offset/bits.
        pub inline fn bitfieldInsert(self: Self, insert: anytype, offset: anytype, bits: anytype) Self {
            var r: storage_type = undefined;
            inline for (0..L) |i| r[i] = scalar.bitfieldInsert(
                self.v[i],
                if (comptime isVec(@TypeOf(insert))) insert.v[i] else insert,
                if (comptime isVec(@TypeOf(offset))) offset.v[i] else offset,
                if (comptime isVec(@TypeOf(bits))) bits.v[i] else bits,
            );
            return .{ .v = r };
        }

        /// Per-component bit reversal (GLM `bitfieldReverse`): mirror the
        /// whole lane, e.g. `0b1101 → 0b1011` for 4-bit lanes. Particularly
        /// useful for hashing grid hash cells across lanes.
        pub inline fn bitfieldReverse(self: Self) Self {
            return self.apply(scalar.bitfieldReverse);
        }

        /// Per-component unsigned addition with carry out (GLM
        /// `uaddCarry`): `sum = x + y` modulo 2^bits, `carry` is 1 where
        /// the addition overflowed. Enables multi-precision arithmetic,
        /// e.g. a 128-bit accumulator in 4 lanes.
        pub fn uaddCarry(self: Self, rhs: Self) struct { sum: Self, carry: Self } {
            var s: storage_type = undefined;
            var c: storage_type = undefined;
            inline for (0..L) |i| {
                const r = @addWithOverflow(self.v[i], rhs.v[i]);
                s[i] = r[0];
                c[i] = if (r[1]) scalar.cast(T, 1) else scalar.cast(T, 0);
            }
            return .{ .sum = .{ .v = s }, .carry = .{ .v = c } };
        }

        /// Per-component unsigned subtraction with borrow out (GLM
        /// `usubBorrow`): `diff = x - y` modulo 2^bits, `borrow` is 1
        /// where the subtraction underflowed. The unsigned twin of
        /// `uaddCarry`.
        pub fn usubBorrow(self: Self, rhs: Self) struct { diff: Self, borrow: Self } {
            var d: storage_type = undefined;
            var b: storage_type = undefined;
            inline for (0..L) |i| {
                const r = @subWithOverflow(self.v[i], rhs.v[i]);
                d[i] = r[0];
                b[i] = if (r[1]) scalar.cast(T, 1) else scalar.cast(T, 0);
            }
            return .{ .diff = .{ .v = d }, .borrow = .{ .v = b } };
        }

        // ---- ext/vector_common (GLM 1.1 additions) ----

        /// Component-wise NaN-safe minimum (GLM `fmin`): unlike `min`, a NaN
        /// operand is ignored, so `fmin` with a sentinel NaN "clamps" a
        /// value into a valid range: `v.fmin(valid_limit)`.
        pub inline fn fmin(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.fmin);
        }

        /// Three-way NaN-safe minimum (GLM `fmin(x, y, z)`).
        pub inline fn fmin3(self: Self, b: anytype, c: anytype) Self {
            return self.apply3(b, c, scalar.fmin3);
        }

        /// Four-way NaN-safe minimum (GLM `fmin(x, y, z, w)`).
        pub inline fn fmin4(self: Self, b: anytype, c: anytype, d: anytype) Self {
            return self.fmin3(b, c).fmin(d);
        }

        /// Component-wise NaN-safe maximum (GLM `fmax`): NaN operands are
        /// ignored, so `fmax` with a NaN removes bad lanes from simulation
        /// data instead of poisoning it (unlike `max`).
        pub inline fn fmax(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.fmax);
        }

        /// Three-way NaN-safe maximum (GLM `fmax(x, y, z)`).
        pub inline fn fmax3(self: Self, b: anytype, c: anytype) Self {
            return self.apply3(b, c, scalar.fmax3);
        }

        /// Four-way NaN-safe maximum (GLM `fmax(x, y, z, w)`).
        pub inline fn fmax4(self: Self, b: anytype, c: anytype, d: anytype) Self {
            return self.fmax3(b, c).fmax(d);
        }

        /// Component-wise clamp that ignores NaN (GLM `fclamp(x, lo, hi)`):
        /// each lane is pinned into [lo, hi], but a NaN lane passes through
        /// untouched. Use on data feeds where NaN marks "missing".
        pub inline fn fclamp(self: Self, lo: anytype, hi: anytype) Self {
            return self.apply3(lo, hi, scalar.fclamp);
        }

        /// Clamp every component into [0, 1] (GLM `clamp01`, 1.1 addition).
        /// Shorthand for `clamp(0, 1)`, common in color math.
        pub inline fn clamp01(self: Self) Self {
            return self.apply(scalar.clamp01);
        }

        /// Fractional part of every component (GLM `repeat`, 1.1 addition,
        /// identical to `fract`). Think of it as wrapping a 1D coordinate
        /// into the unit interval and a 2D texture coordinate into [0,1)².
        pub inline fn repeat(self: Self) Self {
            return self.fract();
        }

        /// Reflect-and-repeat per component (GLM `mirrorClamp`, 1.1
        /// addition): `|x| mod 1`, i.e. a sawtooth clamped to [-1, 1] first.
        /// Use for mirrored, seamless tiling, e.g. a brick wall UV.
        pub inline fn mirrorClamp(self: Self) Self {
            return self.abs().fract();
        }

        /// Mirrored repetition per component (GLM `mirrorRepeat`, 1.1
        /// addition): triangle wave from 0 to 1, `mirrorRepeat(t)` wraps a
        /// time-line like a ping-pong loop without a discontinuity in the
        /// derivative at the fold (unlike `fract`).
        pub fn mirrorRepeat(self: Self) Self {
            var r: storage_type = undefined;
            inline for (0..L) |i| r[i] = scalar.mirrorRepeat(self.v[i]);
            return .{ .v = r };
        }

        /// Round away from zero, returning an i32 vector (GLM `iround`, 1.1
        /// addition). Use when a float must become an array index or pixel
        /// coordinate.
        pub inline fn iround(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.iround(self.v[i]);
            return .{ .v = r };
        }

        /// Round away from zero, returning a u32 vector (GLM `uround`, 1.1
        /// addition); negative results wrap into the unsigned domain via
        /// float→uint cast semantics, so prefer `iround` for positions.
        pub inline fn uround(self: Self) Vec(L, u32) {
            var r: @Vector(L, u32) = undefined;
            inline for (0..L) |i| r[i] = scalar.uround(self.v[i]);
            return .{ .v = r };
        }

        /// Three-way component-wise minimum (GLM `min(x, y, z)`); thriftier
        /// than chaining two `min` calls into temporaries.
        pub inline fn min3(self: Self, b: Self, c: Self) Self {
            return self.min(b).min(c);
        }

        /// Four-way component-wise minimum (GLM `min(x, y, z, w)`).
        pub inline fn min4(self: Self, b: Self, c: Self, d: Self) Self {
            return self.min(b).min(c).min(d);
        }

        /// Three-way component-wise maximum (GLM `max(x, y, z)`).
        pub inline fn max3(self: Self, b: Self, c: Self) Self {
            return self.max(b).max(c);
        }

        /// Four-way component-wise maximum (GLM `max(x, y, z, w)`).
        pub inline fn max4(self: Self, b: Self, c: Self, d: Self) Self {
            return self.max(b).max(c).max(d);
        }

        // ---- ext/vector_reciprocal ----

        /// Per-component secant (GLM `sec`): `1/cos(x)`. Undefined where
        /// cosine vanishes. All six trig reciprocals below behave like
        /// their GLM counterparts and are NaN when the base function is 0.
        pub inline fn sec(self: Self) Self {
            return self.apply(scalar.sec);
        }

        /// Per-component cosecant (GLM `csc`): `1/sin(x)`.
        pub inline fn csc(self: Self) Self {
            return self.apply(scalar.csc);
        }

        /// Per-component cotangent (GLM `cot`): `cos(x)/sin(x)`.
        pub inline fn cot(self: Self) Self {
            return self.apply(scalar.cot);
        }

        /// Per-component arc secant (GLM `asec`): `acos(1/x)`, domain
        /// |x| ≥ 1.
        pub inline fn asec(self: Self) Self {
            return self.apply(scalar.asec);
        }

        /// Per-component arc cosecant (GLM `acsc`): `asin(1/x)`, domain
        /// |x| ≥ 1.
        pub inline fn acsc(self: Self) Self {
            return self.apply(scalar.acsc);
        }

        /// Per-component arc cotangent (GLM `acot`): `atan(1/x)`.
        /// `acot(1) == π/4`.
        pub inline fn acot(self: Self) Self {
            return self.apply(scalar.acot);
        }

        /// Per-component hyperbolic secant (GLM `sech`): `1/cosh(x)`.
        pub inline fn sech(self: Self) Self {
            return self.apply(scalar.sech);
        }

        /// Per-component hyperbolic cosecant (GLM `csch`): `1/sinh(x)`;
        /// NaN where the hyperbolic sine is zero.
        pub inline fn csch(self: Self) Self {
            return self.apply(scalar.csch);
        }

        /// Per-component hyperbolic cotangent (GLM `coth`):
        /// `cosh(x)/sinh(x)`; NaN at x = 0.
        pub inline fn coth(self: Self) Self {
            return self.apply(scalar.coth);
        }

        /// Per-component inverse hyperbolic secant (GLM `asech`):
        /// `acosh(1/x)`, domain (0, 1].
        pub inline fn asech(self: Self) Self {
            return self.apply(scalar.asech);
        }

        /// Per-component inverse hyperbolic cosecant (GLM `acsch`):
        /// `asinh(1/x)`, defined for x ≠ 0.
        pub inline fn acsch(self: Self) Self {
            return self.apply(scalar.acsch);
        }

        /// Per-component inverse hyperbolic cotangent (GLM `acoth`):
        /// `atanh(1/x)`, domain |x| > 1.
        pub inline fn acoth(self: Self) Self {
            return self.apply(scalar.acoth);
        }

        // ---- ext/vector_relational + gtc/epsilon ----

        /// Per-component tolerance comparison with an absolute epsilon
        /// (GLM `equalEps(x, y, eps)`): lane true iff `|x − y| < eps`.
        /// `eps` may be a vector or scalar. Use instead of `==` on float
        /// data — integration results are rarely bit-exact.
        pub fn equalEps(self: Self, rhs: anytype, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.equalEps(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        /// Per-component negation of `equalEps` (GLM
        /// `notEqualEps(x, y, eps)`): lane true iff `|x − y| ≥ eps`.
        pub fn notEqualEps(self: Self, rhs: anytype, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.notEqualEps(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        /// Per-component `|x − y| < eps` (GLM `epsilonEqual`). Historically
        /// GLM's epsilon versions used a relative-tolerance trick; here the
        /// behavior matches `equalEps` — an absolute comparison against
        /// `eps` with user-controlled magnitude (GLM defaults to 0.1 for
        /// floats).
        pub fn epsilonEqual(self: Self, rhs: Self, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.epsilonEqual(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        /// Per-component `|x − y| ≥ eps` (GLM `epsilonNotEqual`); the
        /// negation of `epsilonEqual`.
        pub fn epsilonNotEqual(self: Self, rhs: Self, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.epsilonNotEqual(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        /// Per-component ULP comparison (GLM/GLSL `equal(x, y, ulps)`):
        /// lane true iff the float bit patterns differ by at most
        /// `max_ulps` ULPs, treating transcendental results as equal even
        /// when they are not bit-identical. Unlike absolute epsilons this
        /// stays meaningful across exponent scales (1e-20 and 1e20).
        pub fn equalULP(self: Self, rhs: Self, max_ulps: anytype) Vec(L, bool) {
            const UT = @TypeOf(max_ulps);
            const uv = comptime isVec(UT);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                const u: i32 = if (uv) max_ulps.v[i] else max_ulps;
                const T2 = @TypeOf(self.v[i]);
                if (T2 == f64) {
                    const a: i64 = @bitCast(self.v[i]);
                    const b: i64 = @bitCast(rhs.v[i]);
                    if ((a < 0) != (b < 0)) {
                        const mant_a = a & ((@as(i64, 1) << 52) - 1);
                        const mant_b = b & ((@as(i64, 1) << 52) - 1);
                        const exp_a = (a >> 52) & ((@as(i64, 1) << 11) - 1);
                        const exp_b = (b >> 52) & ((@as(i64, 1) << 11) - 1);
                        r[i] = mant_a == mant_b and exp_a == exp_b;
                    } else {
                        r[i] = scalar.abs(a - b) <= @as(i64, u);
                    }
                } else {
                    const a: i32 = @bitCast(@as(f32, self.v[i]));
                    const b: i32 = @bitCast(@as(f32, rhs.v[i]));
                    if ((a < 0) != (b < 0)) {
                        const mant_a = a & ((@as(i32, 1) << 23) - 1);
                        const mant_b = b & ((@as(i32, 1) << 23) - 1);
                        const exp_a = (a >> 23) & ((@as(i32, 1) << 8) - 1);
                        const exp_b = (b >> 23) & ((@as(i32, 1) << 8) - 1);
                        r[i] = mant_a == mant_b and exp_a == exp_b;
                    } else {
                        r[i] = scalar.abs(a - b) <= u;
                    }
                }
            }
            return .{ .v = r };
        }

        /// Per-component negation of `equalULP`: lane true iff the two
        /// floats are more than `max_ulps` ULPs apart.
        pub fn notEqualULP(self: Self, rhs: Self, max_ulps: anytype) Vec(L, bool) {
            const e = self.equalULP(rhs, max_ulps);
            return e.not_();
        }

        /// Component-wise logical NOT on a bool vector (GLM `not_`):
        /// flips every lane. Combined with the relational methods it forms
        /// the building block for masks: `v.notEqual(rhs).not_().any()`.
        pub fn not_(self: Self) Vec(L, bool) {
            if (comptime T != bool) @compileError("not_() requires a bool vector");
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = !self.v[i];
            return .{ .v = r };
        }

        // ---- ext/vector_ulp ----

        /// Per-component next representable float, toward +∞ (GLM
        /// `nextFloat`, 1.1 addition): the smallest float strictly greater
        /// than the lane, or the lane itself at +∞. Use to step a float
        /// through every representable value in a scan, e.g. to find the
        /// exact ULP count between two values.
        pub inline fn nextFloat(self: Self) Self {
            return self.apply(scalar.nextFloat);
        }

        /// Per-component previous representable float, toward −∞ (GLM
        /// `prevFloat`, 1.1 addition): the largest float strictly less than
        /// the lane, or the lane itself at −∞.
        pub inline fn prevFloat(self: Self) Self {
            return self.apply(scalar.prevFloat);
        }

        /// Per-component count of representable floats strictly between the
        /// two operands (GLM `floatDistance`, 1.1 addition), signed by the
        /// direction from `self` toward `rhs` and returned as i64. This is
        /// the exact "distance in ULPs" that `equalULP` approximates.
        pub inline fn floatDistance(self: Self, rhs: Self) Vec(L, i64) {
            var r: @Vector(L, i64) = undefined;
            inline for (0..L) |i| r[i] = scalar.floatDistance(self.v[i], rhs.v[i]);
            return .{ .v = r };
        }

        // ---- bit-casts (func_common) ----

        /// Reinterpret the float bits of each lane as an i32 (GLM
        /// `floatBitsToInt`, GLSL `floatBitsToInt`). Use for exact
        /// comparisons, hashing, or packing into integers — the bit pattern
        /// of a float is preserved, including NaN payloads, and the result
        /// for -0.0 differs from +0.0 (`0x80000000` vs `0x00000000`).
        pub inline fn floatBitsToInt(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.floatBitsToInt(@floatCast(self.v[i]));
            return .{ .v = r };
        }

        /// Reinterpret the float bits of each lane as a u32 (GLM
        /// `floatBitsToUint`). The unsigned twin of `floatBitsToInt`.
        pub inline fn floatBitsToUint(self: Self) Vec(L, u32) {
            var r: @Vector(L, u32) = undefined;
            inline for (0..L) |i| r[i] = scalar.floatBitsToUint(@floatCast(self.v[i]));
            return .{ .v = r };
        }

        /// Reinterpret i32 bits as f32 (GLM `intBitsToFloat`); the inverse
        /// of `floatBitsToInt`. Use to decode float data that was stored in
        /// integer storage, or to construct floats from raw IEEE patterns.
        pub inline fn intBitsToFloat(self: Self) Vec(L, f32) {
            var r: @Vector(L, f32) = undefined;
            inline for (0..L) |i| r[i] = scalar.intBitsToFloat(self.v[i]);
            return .{ .v = r };
        }

        /// Reinterpret u32 bits as f32 (GLM `uintBitsToFloat`); the inverse
        /// of `floatBitsToUint`.
        pub inline fn uintBitsToFloat(self: Self) Vec(L, f32) {
            var r: @Vector(L, f32) = undefined;
            inline for (0..L) |i| r[i] = scalar.uintBitsToFloat(self.v[i]);
            return .{ .v = r };
        }

        // ---- ext/vector_integer + gtc/round ----

        /// Per-component power-of-two test (GLM `isPowerOfTwo`, GLSL
        /// `isPowerOfTwo`): lane true iff the integer has exactly one set
        /// bit. Useful after `floorPowerOfTwo` to detect that an allocation
        /// or atlas size is already a valid power of two.
        pub inline fn isPowerOfTwo(self: Self) Vec(L, bool) {
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = scalar.isPowerOfTwo(self.v[i]);
            return .{ .v = r };
        }

        /// Per-component "is a multiple of `multiple`" test (GLM
        /// `isMultiple`): lane true iff `x mod m == 0`. Handles signed
        /// operands by comparing absolute values, so `isMultiple(-8, 4)`
        /// is also true, while 0 is a multiple of anything.
        pub inline fn isMultiple(self: Self, multiple: anytype) Vec(L, bool) {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.isMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        /// Next integer ≥ `self[i]` divisible by `multiple` (GLM
        /// `nextMultiple`); `multiple` may be a vector or scalar. GLM
        /// requires a strictly positive `multiple`. Use to round a size up
        /// to an alignment, e.g. a buffer stride.
        pub fn nextMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.nextMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        /// Previous integer ≤ `self[i]` divisible by `multiple` (GLM
        /// `prevMultiple`); the floor counterpart of `nextMultiple`.
        pub fn prevMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.prevMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        /// Any integer ≥ `self[i]` divisible by `multiple` that minimizes
        /// the distance to `self[i]` (GLM `ceilMultiple` — despite the
        /// name its behavior is nearest-aligned with ties going up);
        /// `multiple` may be a vector or scalar.
        pub fn ceilMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.ceilMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        /// Any integer ≤ `self[i]` divisible by `multiple` that minimizes
        /// the distance to `self[i]` (GLM `floorMultiple` — nearest-aligned
        /// with ties going down). The twin of `ceilMultiple`.
        pub fn floorMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.floorMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        /// Any integer divisible by `multiple` that minimizes the distance
        /// to `self[i]` (GLM `roundMultiple` — nearest-aligned, ties down).
        pub fn roundMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.roundMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        /// Smallest power of two ≥ each lane (GLM `ceilPowerOfTwo`).
        /// Classic use: size a render target or texture to a POT dimension.
        pub fn ceilPowerOfTwo(self: Self) Self {
            return self.apply(scalar.ceilPowerOfTwo);
        }

        /// Largest power of two ≤ each lane (GLM `floorPowerOfTwo`).
        /// Good for quantizing sizes down to a cache-line-friendly block.
        pub fn floorPowerOfTwo(self: Self) Self {
            return self.apply(scalar.floorPowerOfTwo);
        }

        /// Power of two nearest to each lane (GLM `roundPowerOfTwo`).
        pub fn roundPowerOfTwo(self: Self) Self {
            return self.apply(scalar.roundPowerOfTwo);
        }

        /// Smallest power of two ≥ each lane, returning ≥ 1 (GLM
        /// `nextPowerOfTwo`): an alias of `ceilPowerOfTwo` that instead of
        /// returning 0 for a 0/negative input clamps the result to 1 —
        /// presence check `nextPowerOfTwo(0) == 1` holds.
        pub fn nextPowerOfTwo(self: Self) Self {
            return self.apply(scalar.nextPowerOfTwo);
        }

        /// Largest power of two ≤ each lane, clamping results to 1 for
        /// inputs below 2 (GLM `prevPowerOfTwo`): alias of
        /// `floorPowerOfTwo` except that 0 and 1 (and negatives) yield 1
        /// instead of 0.
        pub fn prevPowerOfTwo(self: Self) Self {
            return self.apply(scalar.prevPowerOfTwo);
        }

        /// Per-component index of the start of the most significant run of
        /// `count` consecutive set bits (GLM `findNSB`), or -1 when no such
        /// run fits. Use for slot allocators: with `count` = freelist size
        /// it finds the first spot that can host a run.
        pub fn findNSB(self: Self, count: anytype) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.findNSB(self.v[i], count);
            return .{ .v = r };
        }

        // ---- reductions ----

        /// Sum of all components (GLM `compAdd`, GLSL `compAdd`): a scalar.
        /// For dot products use `dot`, which is the same without the extra
        /// allocation.
        pub inline fn compAdd(self: Self) T {
            return @reduce(.Add, self.v);
        }

        /// Product of all components (GLM `compMul`): `x*y*z*w`. Zero in
        /// any lane zeroes the result — check before dividing by it.
        pub inline fn compMul(self: Self) T {
            return @reduce(.Mul, self.v);
        }

        /// Minimum of all components (GLM `compMin`). Useful to find the
        /// tightest bounding extent of a vector set.
        pub inline fn compMin(self: Self) T {
            return @reduce(.Min, self.v);
        }

        /// Maximum of all components (GLM `compMax`). With `compMin` this
        /// brackets a value set like an AABB.
        pub inline fn compMax(self: Self) T {
            return @reduce(.Max, self.v);
        }

        // ---- printing ----

        /// `{any}` formatter: prints the vector as `{x,y,z}` with `{d}`
        /// float formatting (eg `{1.5,2,3}`); bool vectors print
        /// `{true,false}`. Enables `std.debug.print("{any}", .{v})` and
        /// `{any}` in formatted logs.
        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.writeByte('{');
            for (0..L) |i| {
                if (i != 0) try writer.writeByte(',');
                if (comptime T == bool)
                    try writer.print("{}", .{self.v[i]})
                else
                    try writer.print("{d}", .{self.v[i]});
            }
            try writer.writeByte('}');
        }
    };
}

fn comptimePrint(comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.comptimePrint(fmt, args);
}
