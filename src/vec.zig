//! Vector math — GLM-equivalent `vec<L, T>` implementation.
//! All operations are methods on the struct, matching GLM's free functions
//! with the receiver as the first argument (e.g. `a.dot(b)` == `glm::dot(a, b)`).

const std = @import("std");
const scalar = @import("scalar.zig");

pub fn isVec(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "len") and @hasField(T, "v");
}

const floatType = scalar.floatType;

pub fn Vec(comptime L: usize, comptime T: type) type {
    return struct {
        pub const Self = @This();
        pub const len: comptime_int = L;
        pub const value_type: type = T;
        pub const storage_type: type = @Vector(L, T);
        pub const float_type: type = floatType(T);

        v: storage_type,

        // ---- constructors ----

        pub inline fn zero() Self {
            return .{ .v = @splat(scalar.cast(T, 0)) };
        }

        pub inline fn one() Self {
            return .{ .v = @splat(scalar.cast(T, 1)) };
        }

        pub inline fn fill(v: anytype) Self {
            return .{ .v = @splat(scalar.cast(T, v)) };
        }

        pub inline fn unit(comptime i: usize) Self {
            var r: storage_type = @splat(scalar.cast(T, 0));
            r[i] = scalar.cast(T, 1);
            return .{ .v = r };
        }

        /// GLM-style constructor: takes a tuple of scalars and/or smaller vectors,
        /// a scalar (splat), or another vector (truncate/copy).
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

        pub inline fn get(self: Self, i: usize) T {
            const p: *const storage_type = &self.v;
            return p[i];
        }

        pub inline fn set(self: *Self, i: usize, val: T) void {
            self.v[i] = val;
        }

        pub inline fn x(self: Self) T {
            return self.v[0];
        }

        pub inline fn y(self: Self) T {
            if (comptime L < 2) @compileError("vector has no y component");
            return self.v[1];
        }

        pub inline fn z(self: Self) T {
            if (comptime L < 3) @compileError("vector has no z component");
            return self.v[2];
        }

        pub inline fn w(self: Self) T {
            if (comptime L < 4) @compileError("vector has no w component");
            return self.v[3];
        }

        pub inline fn setX(self: *Self, val: T) void {
            self.v[0] = val;
        }

        pub inline fn setY(self: *Self, val: T) void {
            if (comptime L < 2) @compileError("vector has no y component");
            self.v[1] = val;
        }

        pub inline fn setZ(self: *Self, val: T) void {
            if (comptime L < 3) @compileError("vector has no z component");
            self.v[2] = val;
        }

        pub inline fn setW(self: *Self, val: T) void {
            if (comptime L < 4) @compileError("vector has no w component");
            self.v[3] = val;
        }

        // ---- swizzles ----

        inline fn swz2(self: Self, comptime a: usize, comptime b: usize) Vec(2, T) {
            return .{ .v = @Vector(2, T){ self.v[a], self.v[b] } };
        }

        inline fn swz3(self: Self, comptime a: usize, comptime b: usize, comptime c: usize) Vec(3, T) {
            return .{ .v = @Vector(3, T){ self.v[a], self.v[b], self.v[c] } };
        }

        pub inline fn xy(self: Self) Vec(2, T) {
            if (comptime L < 2) @compileError("swizzle out of range");
            return self.swz2(0, 1);
        }

        pub inline fn xz(self: Self) Vec(2, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz2(0, 2);
        }

        pub inline fn yz(self: Self) Vec(2, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz2(1, 2);
        }

        pub inline fn xw(self: Self) Vec(2, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return self.swz2(0, 3);
        }

        pub inline fn yw(self: Self) Vec(2, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return self.swz2(1, 3);
        }

        pub inline fn zw(self: Self) Vec(2, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return self.swz2(2, 3);
        }

        pub inline fn xyz(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(0, 1, 2);
        }

        pub inline fn xzy(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(0, 2, 1);
        }

        pub inline fn yxz(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(1, 0, 2);
        }

        pub inline fn yzx(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(1, 2, 0);
        }

        pub inline fn zxy(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(2, 0, 1);
        }

        pub inline fn zyx(self: Self) Vec(3, T) {
            if (comptime L < 3) @compileError("swizzle out of range");
            return self.swz3(2, 1, 0);
        }

        pub inline fn xyzw(self: Self) Vec(4, T) {
            if (comptime L < 4) @compileError("swizzle out of range");
            return .{ .v = @Vector(4, T){ self.v[0], self.v[1], self.v[2], self.v[3] } };
        }

        // ---- element-wise application helpers ----

        fn apply(self: Self, comptime f: anytype) Self {
            var r: storage_type = undefined;
            inline for (0..L) |i| r[i] = f(self.v[i]);
            return .{ .v = r };
        }

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

        pub inline fn add(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v + rhs.v };
            return .{ .v = self.v + @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        pub inline fn sub(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v - rhs.v };
            return .{ .v = self.v - @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        pub inline fn mul(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v * rhs.v };
            return .{ .v = self.v * @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        pub inline fn div(self: Self, rhs: anytype) Self {
            if (comptime isVec(@TypeOf(rhs))) return .{ .v = self.v / rhs.v };
            return .{ .v = self.v / @as(storage_type, @splat(scalar.cast(T, rhs))) };
        }

        pub inline fn neg(self: Self) Self {
            return .{ .v = -self.v };
        }

        pub inline fn mod(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.mod);
        }

        /// Component-wise reciprocal (GLM `inverse(vec)`).
        pub inline fn inverse(self: Self) Self {
            return .{ .v = @as(storage_type, @splat(scalar.cast(T, 1))) / self.v };
        }

        // ---- relational ----

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

        pub inline fn lessThan(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .lt);
        }

        pub inline fn lessThanEqual(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .le);
        }

        pub inline fn greaterThan(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .gt);
        }

        pub inline fn greaterThanEqual(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .ge);
        }

        pub inline fn equal(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .eq);
        }

        pub inline fn notEqual(self: Self, rhs: anytype) Vec(L, bool) {
            return self.cmp(rhs, .ne);
        }

        // ---- bool-vector reductions ----

        pub inline fn any(self: Self) bool {
            if (comptime T != bool) @compileError("any() requires a bool vector");
            return @reduce(.Or, self.v);
        }

        pub inline fn all(self: Self) bool {
            if (comptime T != bool) @compileError("all() requires a bool vector");
            return @reduce(.And, self.v);
        }

        // ---- common ----

        pub inline fn abs(self: Self) Self {
            return self.apply(scalar.abs);
        }

        pub inline fn sign(self: Self) Self {
            return self.apply(scalar.sign);
        }

        pub inline fn floor(self: Self) Self {
            return self.apply(scalar.floor);
        }

        pub inline fn ceil(self: Self) Self {
            return self.apply(scalar.ceil);
        }

        pub inline fn round(self: Self) Self {
            return self.apply(scalar.round);
        }

        pub inline fn roundEven(self: Self) Self {
            return self.apply(scalar.roundEven);
        }

        pub inline fn trunc(self: Self) Self {
            return self.apply(scalar.trunc);
        }

        pub inline fn fract(self: Self) Self {
            return self.apply(scalar.fract);
        }

        pub inline fn min(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.min);
        }

        pub inline fn max(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.max);
        }

        pub inline fn clamp(self: Self, lo: anytype, hi: anytype) Self {
            return self.apply3(lo, hi, clampHelper);
        }

        fn clampHelper(lo: anytype, hi: anytype, v: anytype) @TypeOf(v) {
            return scalar.clamp(v, lo, hi);
        }

        /// `mix(rhs, a)`: a may be a vector, scalar or bool.
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

        /// `step(edge)` == GLM `step(edge, self)`.
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

        /// `smoothstep(e0, e1)` == GLM `smoothstep(e0, e1, self)`.
        pub inline fn smoothstep(self: Self, e0: anytype, e1: anytype) Self {
            return self.apply3(e0, e1, scalar.smoothstep);
        }

        pub inline fn fma(self: Self, rhs: anytype, c: anytype) Self {
            const RT = @TypeOf(rhs);
            const CT = @TypeOf(c);
            if (comptime isVec(RT) and isVec(CT))
                return .{ .v = @mulAdd(storage_type, self.v, rhs.v, c.v) };
            return self.apply3(rhs, c, scalar.fma);
        }

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

        pub fn isNan(self: Self) Vec(L, bool) {
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = scalar.isNan(self.v[i]);
            return .{ .v = r };
        }

        pub fn isInf(self: Self) Vec(L, bool) {
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = scalar.isInf(self.v[i]);
            return .{ .v = r };
        }

        // ---- trigonometric ----

        pub inline fn radians(self: Self) Self {
            return self.apply(scalar.radians);
        }

        pub inline fn degrees(self: Self) Self {
            return self.apply(scalar.degrees);
        }

        pub inline fn sin(self: Self) Self {
            return self.apply(scalar.sin);
        }

        pub inline fn cos(self: Self) Self {
            return self.apply(scalar.cos);
        }

        pub inline fn tan(self: Self) Self {
            return self.apply(scalar.tan);
        }

        pub inline fn asin(self: Self) Self {
            return self.apply(scalar.asin);
        }

        pub inline fn acos(self: Self) Self {
            return self.apply(scalar.acos);
        }

        pub inline fn atan(self: Self) Self {
            return self.apply(scalar.atan);
        }

        /// `atan2(y)` == GLM `atan2(self, y)`.
        pub inline fn atan2(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.atan2);
        }

        pub inline fn sinh(self: Self) Self {
            return self.apply(scalar.sinh);
        }

        pub inline fn cosh(self: Self) Self {
            return self.apply(scalar.cosh);
        }

        pub inline fn tanh(self: Self) Self {
            return self.apply(scalar.tanh);
        }

        pub inline fn asinh(self: Self) Self {
            return self.apply(scalar.asinh);
        }

        pub inline fn acosh(self: Self) Self {
            return self.apply(scalar.acosh);
        }

        pub inline fn atanh(self: Self) Self {
            return self.apply(scalar.atanh);
        }

        // ---- exponential ----

        pub inline fn pow(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.pow);
        }

        pub inline fn exp(self: Self) Self {
            return self.apply(scalar.exp);
        }

        pub inline fn exp2(self: Self) Self {
            return self.apply(scalar.exp2);
        }

        pub inline fn log(self: Self) Self {
            return self.apply(scalar.log);
        }

        pub inline fn log2(self: Self) Self {
            return self.apply(scalar.log2);
        }

        pub inline fn sqrt(self: Self) Self {
            return self.apply(scalar.sqrt);
        }

        pub inline fn inversesqrt(self: Self) Self {
            return self.apply(scalar.inversesqrt);
        }

        // ---- geometric ----

        pub inline fn dot(self: Self, b: Self) T {
            return @reduce(.Add, self.v * b.v);
        }

        pub fn length(self: Self) float_type {
            const d = @reduce(.Add, self.v * self.v);
            return scalar.sqrt(scalar.cast(float_type, d));
        }

        pub inline fn distance(self: Self, b: Self) float_type {
            return self.sub(b).length();
        }

        pub fn cross(self: Self, b: Self) Self {
            if (comptime L != 3) @compileError("cross is only defined for 3-component vectors");
            return .{ .v = @Vector(3, T){
                self.v[1] * b.v[2] - self.v[2] * b.v[1],
                self.v[2] * b.v[0] - self.v[0] * b.v[2],
                self.v[0] * b.v[1] - self.v[1] * b.v[0],
            } };
        }

        pub fn normalize(self: Self) Vec(L, float_type) {
            const d = @reduce(.Add, self.v * self.v);
            const is: float_type = scalar.inversesqrt(scalar.cast(float_type, d));
            const c: @Vector(L, float_type) = if (T == float_type)
                self.v
            else
                @floatCast(self.v);
            return .{ .v = c * @as(@Vector(L, float_type), @splat(is)) };
        }

        /// `faceforward(i, nref)` == GLM `faceforward(self, i, nref)` (self is N).
        pub inline fn faceforward(self: Self, i: Self, nref: Self) Self {
            return if (nref.dot(i) < 0) self else self.neg();
        }

        /// `reflect(n)` == GLM `reflect(self, n)` (self is I).
        pub inline fn reflect(self: Self, n: Self) Self {
            return self.sub(n.mul(self.dot(n)).mul(@as(T, 2)));
        }

        /// `refract(n, eta)` == GLM `refract(self, n, eta)` (self is I).
        pub fn refract(self: Self, n: Self, eta: anytype) Self {
            const et: T = scalar.cast(T, eta);
            const dot_value = self.dot(n);
            const k = @as(T, 1) - et * et * (@as(T, 1) - dot_value * dot_value);
            if (k < 0) return Self.zero();
            return self.mul(et).sub(n.mul(scalar.sqrt(k) + et * dot_value));
        }

        // ---- integer / bit ----

        pub inline fn bitCount(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.bitCount(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn findLSB(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.findLSB(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn findMSB(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.findMSB(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn bitfieldExtract(self: Self, offset: anytype, bits: anytype) Self {
            var r: storage_type = undefined;
            if (comptime isVec(@TypeOf(offset))) {
                inline for (0..L) |i| r[i] = scalar.bitfieldExtract(self.v[i], offset.v[i], bits.v[i]);
            } else {
                inline for (0..L) |i| r[i] = scalar.bitfieldExtract(self.v[i], offset, bits);
            }
            return .{ .v = r };
        }

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

        pub inline fn bitfieldReverse(self: Self) Self {
            return self.apply(scalar.bitfieldReverse);
        }

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

        pub inline fn fmin(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.fmin);
        }

        pub inline fn fmin3(self: Self, b: anytype, c: anytype) Self {
            return self.apply3(b, c, scalar.fmin3);
        }

        pub inline fn fmin4(self: Self, b: anytype, c: anytype, d: anytype) Self {
            return self.fmin3(b, c).fmin(d);
        }

        pub inline fn fmax(self: Self, rhs: anytype) Self {
            return self.apply2(rhs, scalar.fmax);
        }

        pub inline fn fmax3(self: Self, b: anytype, c: anytype) Self {
            return self.apply3(b, c, scalar.fmax3);
        }

        pub inline fn fmax4(self: Self, b: anytype, c: anytype, d: anytype) Self {
            return self.fmax3(b, c).fmax(d);
        }

        pub inline fn fclamp(self: Self, lo: anytype, hi: anytype) Self {
            return self.apply3(lo, hi, scalar.fclamp);
        }

        pub inline fn clamp01(self: Self) Self {
            return self.apply(scalar.clamp01);
        }

        pub inline fn repeat(self: Self) Self {
            return self.fract();
        }

        pub inline fn mirrorClamp(self: Self) Self {
            return self.abs().fract();
        }

        pub fn mirrorRepeat(self: Self) Self {
            var r: storage_type = undefined;
            inline for (0..L) |i| r[i] = scalar.mirrorRepeat(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn iround(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.iround(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn uround(self: Self) Vec(L, u32) {
            var r: @Vector(L, u32) = undefined;
            inline for (0..L) |i| r[i] = scalar.uround(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn min3(self: Self, b: Self, c: Self) Self {
            return self.min(b).min(c);
        }

        pub inline fn min4(self: Self, b: Self, c: Self, d: Self) Self {
            return self.min(b).min(c).min(d);
        }

        pub inline fn max3(self: Self, b: Self, c: Self) Self {
            return self.max(b).max(c);
        }

        pub inline fn max4(self: Self, b: Self, c: Self, d: Self) Self {
            return self.max(b).max(c).max(d);
        }

        // ---- ext/vector_reciprocal ----

        pub inline fn sec(self: Self) Self {
            return self.apply(scalar.sec);
        }

        pub inline fn csc(self: Self) Self {
            return self.apply(scalar.csc);
        }

        pub inline fn cot(self: Self) Self {
            return self.apply(scalar.cot);
        }

        pub inline fn asec(self: Self) Self {
            return self.apply(scalar.asec);
        }

        pub inline fn acsc(self: Self) Self {
            return self.apply(scalar.acsc);
        }

        pub inline fn acot(self: Self) Self {
            return self.apply(scalar.acot);
        }

        pub inline fn sech(self: Self) Self {
            return self.apply(scalar.sech);
        }

        pub inline fn csch(self: Self) Self {
            return self.apply(scalar.csch);
        }

        pub inline fn coth(self: Self) Self {
            return self.apply(scalar.coth);
        }

        pub inline fn asech(self: Self) Self {
            return self.apply(scalar.asech);
        }

        pub inline fn acsch(self: Self) Self {
            return self.apply(scalar.acsch);
        }

        pub inline fn acoth(self: Self) Self {
            return self.apply(scalar.acoth);
        }

        // ---- ext/vector_relational + gtc/epsilon ----

        pub fn equalEps(self: Self, rhs: anytype, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.equalEps(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        pub fn notEqualEps(self: Self, rhs: anytype, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.notEqualEps(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        pub fn epsilonEqual(self: Self, rhs: Self, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.epsilonEqual(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

        pub fn epsilonNotEqual(self: Self, rhs: Self, eps: anytype) Vec(L, bool) {
            const ET = @TypeOf(eps);
            const ev = comptime isVec(ET);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.epsilonNotEqual(self.v[i], rhs.v[i], if (ev) eps.v[i] else eps);
            }
            return .{ .v = r };
        }

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

        pub fn notEqualULP(self: Self, rhs: Self, max_ulps: anytype) Vec(L, bool) {
            const e = self.equalULP(rhs, max_ulps);
            return e.not_();
        }

        /// GLM `not_`: element-wise logical not of a bool vector.
        pub fn not_(self: Self) Vec(L, bool) {
            if (comptime T != bool) @compileError("not_() requires a bool vector");
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = !self.v[i];
            return .{ .v = r };
        }

        // ---- ext/vector_ulp ----

        pub inline fn nextFloat(self: Self) Self {
            return self.apply(scalar.nextFloat);
        }

        pub inline fn prevFloat(self: Self) Self {
            return self.apply(scalar.prevFloat);
        }

        pub inline fn floatDistance(self: Self, rhs: Self) Vec(L, i64) {
            var r: @Vector(L, i64) = undefined;
            inline for (0..L) |i| r[i] = scalar.floatDistance(self.v[i], rhs.v[i]);
            return .{ .v = r };
        }

        // ---- bit-casts (func_common) ----

        pub inline fn floatBitsToInt(self: Self) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.floatBitsToInt(@floatCast(self.v[i]));
            return .{ .v = r };
        }

        pub inline fn floatBitsToUint(self: Self) Vec(L, u32) {
            var r: @Vector(L, u32) = undefined;
            inline for (0..L) |i| r[i] = scalar.floatBitsToUint(@floatCast(self.v[i]));
            return .{ .v = r };
        }

        pub inline fn intBitsToFloat(self: Self) Vec(L, f32) {
            var r: @Vector(L, f32) = undefined;
            inline for (0..L) |i| r[i] = scalar.intBitsToFloat(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn uintBitsToFloat(self: Self) Vec(L, f32) {
            var r: @Vector(L, f32) = undefined;
            inline for (0..L) |i| r[i] = scalar.uintBitsToFloat(self.v[i]);
            return .{ .v = r };
        }

        // ---- ext/vector_integer + gtc/round ----

        pub inline fn isPowerOfTwo(self: Self) Vec(L, bool) {
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| r[i] = scalar.isPowerOfTwo(self.v[i]);
            return .{ .v = r };
        }

        pub inline fn isMultiple(self: Self, multiple: anytype) Vec(L, bool) {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: @Vector(L, bool) = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.isMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        pub fn nextMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.nextMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        pub fn prevMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.prevMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        pub fn ceilMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.ceilMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        pub fn floorMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.floorMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        pub fn roundMultiple(self: Self, multiple: anytype) Self {
            const MT = @TypeOf(multiple);
            const mv = comptime isVec(MT);
            var r: storage_type = undefined;
            inline for (0..L) |i| {
                r[i] = scalar.roundMultiple(self.v[i], if (mv) multiple.v[i] else multiple);
            }
            return .{ .v = r };
        }

        pub fn ceilPowerOfTwo(self: Self) Self {
            return self.apply(scalar.ceilPowerOfTwo);
        }

        pub fn floorPowerOfTwo(self: Self) Self {
            return self.apply(scalar.floorPowerOfTwo);
        }

        pub fn roundPowerOfTwo(self: Self) Self {
            return self.apply(scalar.roundPowerOfTwo);
        }

        pub fn nextPowerOfTwo(self: Self) Self {
            return self.apply(scalar.nextPowerOfTwo);
        }

        pub fn prevPowerOfTwo(self: Self) Self {
            return self.apply(scalar.prevPowerOfTwo);
        }

        pub fn findNSB(self: Self, count: anytype) Vec(L, i32) {
            var r: @Vector(L, i32) = undefined;
            inline for (0..L) |i| r[i] = scalar.findNSB(self.v[i], count);
            return .{ .v = r };
        }

        // ---- reductions ----

        pub inline fn compAdd(self: Self) T {
            return @reduce(.Add, self.v);
        }

        pub inline fn compMul(self: Self) T {
            return @reduce(.Mul, self.v);
        }

        pub inline fn compMin(self: Self) T {
            return @reduce(.Min, self.v);
        }

        pub inline fn compMax(self: Self) T {
            return @reduce(.Max, self.v);
        }

        // ---- printing ----

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
