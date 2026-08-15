//! Matrix math — GLM-equivalent `mat<C, R, T>` implementation.
//! Column-major storage: `data[c]` is column c, `data[c][r]` is element at
//! (column c, row r), matching GLM's `m[c][r]` indexing.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const Vec = vec.Vec;

pub fn isMat(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "cols") and @hasDecl(T, "rows") and @hasField(T, "data");
}

pub fn Mat(comptime C: usize, comptime R: usize, comptime T: type) type {
    return struct {
        pub const Self = @This();
        pub const cols: comptime_int = C;
        pub const rows: comptime_int = R;
        pub const col_count: comptime_int = C;
        pub const row_count: comptime_int = R;
        pub const value_type: type = T;
        pub const col_type: type = Vec(R, T);
        pub const row_type: type = Vec(C, T);

        data: [C]Vec(R, T),

        // ---- constructors ----

        pub fn zero() Self {
            return .{ .data = [_]Vec(R, T){Vec(R, T).zero()} ** C };
        }

        pub fn identity() Self {
            if (comptime C != R) @compileError("identity requires a square matrix");
            return diag(@as(T, 1));
        }

        /// Matrix with all elements equal to v (GLM elementwise-1 for `1 - a`).
        pub fn one(v: anytype) Self {
            return .{ .data = [_]Vec(R, T){Vec(R, T).fill(scalar.cast(T, v))} ** C };
        }

        pub fn ones() Self {
            return one(@as(T, 1));
        }

        /// Diagonal matrix with value v on the diagonal (GLM `mat4(v)`).
        pub fn diag(v: anytype) Self {
            if (comptime C != R) @compileError("diag requires a square matrix");
            var res = zero();
            inline for (0..C) |i| res.data[i].v[i] = scalar.cast(T, v);
            return res;
        }

        /// Column constructor: tuple of columns (each a Vec(R, T)), or a scalar
        /// which builds a diagonal matrix (GLM `mat(v)`).
        pub fn init(args: anytype) Self {
            const AT = @TypeOf(args);
            if (comptime AT == Self) return args;
            if (comptime scalar.isNumber(AT)) return diag(args);
            const fields = @typeInfo(AT).@"struct".fields;
            var res: Self = undefined;
            comptime var n: usize = 0;
            inline for (fields) |f| {
                const e = @field(args, f.name);
                const ET = @TypeOf(e);
                if (comptime !(ET == Vec(R, T)))
                    @compileError("Mat init: expected column of type " ++ @typeName(Vec(R, T)));
                if (comptime n >= C) @compileError("Mat init: too many columns");
                res.data[n] = e;
                n += 1;
            }
            if (comptime n != C) @compileError("Mat init: expected " ++ comptimePrint("{d}", .{C}) ++ " columns");
            return res;
        }

        // ---- accessors ----

        pub inline fn get(self: Self, c: usize, r: usize) T {
            return self.data[c].v[r];
        }

        pub inline fn set(self: *Self, c: usize, r: usize, v: T) void {
            self.data[c].v[r] = v;
        }

        pub inline fn col(self: Self, i: usize) Vec(R, T) {
            return self.data[i];
        }

        pub fn row(self: Self, i: usize) Vec(C, T) {
            var res: Vec(C, T) = undefined;
            inline for (0..C) |c| res.v[c] = self.data[c].v[i];
            return res;
        }

        // ---- arithmetic ----

        pub fn add(self: Self, m: Self) Self {
            var res: Self = undefined;
            inline for (0..C) |c| res.data[c] = self.data[c].add(m.data[c]);
            return res;
        }

        pub fn sub(self: Self, m: Self) Self {
            var res: Self = undefined;
            inline for (0..C) |c| res.data[c] = self.data[c].sub(m.data[c]);
            return res;
        }

        pub fn mulScalar(self: Self, s: anytype) Self {
            var res: Self = undefined;
            inline for (0..C) |c| res.data[c] = self.data[c].mul(s);
            return res;
        }

        /// GLM `matrixCompMult` — component-wise multiplication.
        pub fn matrixCompMult(self: Self, m: Self) Self {
            var res: Self = undefined;
            inline for (0..C) |c| res.data[c] = self.data[c].mul(m.data[c]);
            return res;
        }

        /// GLM `mat * mat` — column-major matrix product.
        pub fn mul(self: Self, m: anytype) Mat(@TypeOf(m).cols, R, T) {
            const MC = @TypeOf(m).cols;
            var res: Mat(MC, R, T) = undefined;
            inline for (0..MC) |j| res.data[j] = self.mulVec(m.col(j));
            return res;
        }

        /// GLM `mat * vec` — linear combination of columns.
        pub fn mulVec(self: Self, v: Vec(C, T)) Vec(R, T) {
            var res: Vec(R, T) = undefined;
            inline for (0..R) |r| {
                var acc: T = undefined;
                inline for (0..C) |c| {
                    const term = self.data[c].v[r] * v.v[c];
                    acc = if (c == 0) term else acc + term;
                }
                res.v[r] = acc;
            }
            return res;
        }

        /// GLM `transpose`.
        pub fn transpose(self: Self) Mat(R, C, T) {
            var res: Mat(R, C, T) = undefined;
            inline for (0..R) |r| {
                inline for (0..C) |c| res.data[r].v[c] = self.data[c].v[r];
            }
            return res;
        }

        /// GLM `determinant` — square matrices only (2x2, 3x3, 4x4).
        pub fn determinant(self: Self) T {
            if (comptime C != R) @compileError("determinant requires a square matrix");
            if (comptime R == 2) {
                return self.data[0].v[0] * self.data[1].v[1] - self.data[1].v[0] * self.data[0].v[1];
            } else if (comptime R == 3) {
                const t1 = self.data[1].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[1].v[2];
                const t2 = self.data[0].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[0].v[2];
                const t3 = self.data[0].v[1] * self.data[1].v[2] - self.data[1].v[1] * self.data[0].v[2];
                return self.data[0].v[0] * t1 - self.data[1].v[0] * t2 + self.data[2].v[0] * t3;
            } else if (comptime R == 4) {
                const SubFactor00 = self.data[2].v[2] * self.data[3].v[3] - self.data[3].v[2] * self.data[2].v[3];
                const SubFactor01 = self.data[2].v[1] * self.data[3].v[3] - self.data[3].v[1] * self.data[2].v[3];
                const SubFactor02 = self.data[2].v[1] * self.data[3].v[2] - self.data[3].v[1] * self.data[2].v[2];
                const SubFactor03 = self.data[2].v[0] * self.data[3].v[3] - self.data[3].v[0] * self.data[2].v[3];
                const SubFactor04 = self.data[2].v[0] * self.data[3].v[2] - self.data[3].v[0] * self.data[2].v[2];
                const SubFactor05 = self.data[2].v[0] * self.data[3].v[1] - self.data[3].v[0] * self.data[2].v[1];
                const DetCof = Vec(4, T).init(.{
                    (self.data[1].v[1] * SubFactor00 - self.data[1].v[2] * SubFactor01 + self.data[1].v[3] * SubFactor02),
                    -(self.data[1].v[0] * SubFactor00 - self.data[1].v[2] * SubFactor03 + self.data[1].v[3] * SubFactor04),
                    (self.data[1].v[0] * SubFactor01 - self.data[1].v[1] * SubFactor03 + self.data[1].v[3] * SubFactor05),
                    -(self.data[1].v[0] * SubFactor02 - self.data[1].v[1] * SubFactor04 + self.data[1].v[2] * SubFactor05),
                });
                return self.data[0].v[0] * DetCof.v[0] +
                    self.data[0].v[1] * DetCof.v[1] +
                    self.data[0].v[2] * DetCof.v[2] +
                    self.data[0].v[3] * DetCof.v[3];
            } else {
                @compileError("determinant only supported for 2x2, 3x3 and 4x4 matrices");
            }
        }

        /// GLM `inverse` — square matrices only (2x2, 3x3, 4x4).
        pub fn inverse(self: Self) Self {
            if (comptime C != R) @compileError("inverse requires a square matrix");
            if (comptime R == 2) {
                const OneOverDeterminant = @as(T, 1) / (
                    self.data[0].v[0] * self.data[1].v[1] - self.data[1].v[0] * self.data[0].v[1]);
                return init(.{
                    Vec(2, T).init(.{ self.data[1].v[1] * OneOverDeterminant, -self.data[0].v[1] * OneOverDeterminant }),
                    Vec(2, T).init(.{ -self.data[1].v[0] * OneOverDeterminant, self.data[0].v[0] * OneOverDeterminant }),
                });
            } else if (comptime R == 3) {
                const t1 = self.data[1].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[1].v[2];
                const t2 = self.data[0].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[0].v[2];
                const t3 = self.data[0].v[1] * self.data[1].v[2] - self.data[1].v[1] * self.data[0].v[2];
                const OneOverDeterminant = @as(T, 1) / (self.data[0].v[0] * t1 - self.data[1].v[0] * t2 + self.data[2].v[0] * t3);
                var res: Self = undefined;
                res.data[0].v[0] = (self.data[1].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[1].v[2]);
                res.data[1].v[0] = -(self.data[1].v[0] * self.data[2].v[2] - self.data[2].v[0] * self.data[1].v[2]);
                res.data[2].v[0] = (self.data[1].v[0] * self.data[2].v[1] - self.data[2].v[0] * self.data[1].v[1]);
                res.data[0].v[1] = -(self.data[0].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[0].v[2]);
                res.data[1].v[1] = (self.data[0].v[0] * self.data[2].v[2] - self.data[2].v[0] * self.data[0].v[2]);
                res.data[2].v[1] = -(self.data[0].v[0] * self.data[2].v[1] - self.data[2].v[0] * self.data[0].v[1]);
                res.data[0].v[2] = (self.data[0].v[1] * self.data[1].v[2] - self.data[1].v[1] * self.data[0].v[2]);
                res.data[1].v[2] = -(self.data[0].v[0] * self.data[1].v[2] - self.data[1].v[0] * self.data[0].v[2]);
                res.data[2].v[2] = (self.data[0].v[0] * self.data[1].v[1] - self.data[1].v[0] * self.data[0].v[1]);
                return res.mulScalar(OneOverDeterminant);
            } else if (comptime R == 4) {
                const Coef00 = self.data[2].v[2] * self.data[3].v[3] - self.data[3].v[2] * self.data[2].v[3];
                const Coef02 = self.data[1].v[2] * self.data[3].v[3] - self.data[3].v[2] * self.data[1].v[3];
                const Coef03 = self.data[1].v[2] * self.data[2].v[3] - self.data[2].v[2] * self.data[1].v[3];
                const Coef04 = self.data[2].v[1] * self.data[3].v[3] - self.data[3].v[1] * self.data[2].v[3];
                const Coef06 = self.data[1].v[1] * self.data[3].v[3] - self.data[3].v[1] * self.data[1].v[3];
                const Coef07 = self.data[1].v[1] * self.data[2].v[3] - self.data[2].v[1] * self.data[1].v[3];
                const Coef08 = self.data[2].v[1] * self.data[3].v[2] - self.data[3].v[1] * self.data[2].v[2];
                const Coef10 = self.data[1].v[1] * self.data[3].v[2] - self.data[3].v[1] * self.data[1].v[2];
                const Coef11 = self.data[1].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[1].v[2];
                const Coef12 = self.data[2].v[0] * self.data[3].v[3] - self.data[3].v[0] * self.data[2].v[3];
                const Coef14 = self.data[1].v[0] * self.data[3].v[3] - self.data[3].v[0] * self.data[1].v[3];
                const Coef15 = self.data[1].v[0] * self.data[2].v[3] - self.data[2].v[0] * self.data[1].v[3];
                const Coef16 = self.data[2].v[0] * self.data[3].v[2] - self.data[3].v[0] * self.data[2].v[2];
                const Coef18 = self.data[1].v[0] * self.data[3].v[2] - self.data[3].v[0] * self.data[1].v[2];
                const Coef19 = self.data[1].v[0] * self.data[2].v[2] - self.data[2].v[0] * self.data[1].v[2];
                const Coef20 = self.data[2].v[0] * self.data[3].v[1] - self.data[3].v[0] * self.data[2].v[1];
                const Coef22 = self.data[1].v[0] * self.data[3].v[1] - self.data[3].v[0] * self.data[1].v[1];
                const Coef23 = self.data[1].v[0] * self.data[2].v[1] - self.data[2].v[0] * self.data[1].v[1];

                const Fac0 = Vec(4, T).init(.{ Coef00, Coef00, Coef02, Coef03 });
                const Fac1 = Vec(4, T).init(.{ Coef04, Coef04, Coef06, Coef07 });
                const Fac2 = Vec(4, T).init(.{ Coef08, Coef08, Coef10, Coef11 });
                const Fac3 = Vec(4, T).init(.{ Coef12, Coef12, Coef14, Coef15 });
                const Fac4 = Vec(4, T).init(.{ Coef16, Coef16, Coef18, Coef19 });
                const Fac5 = Vec(4, T).init(.{ Coef20, Coef20, Coef22, Coef23 });

                const Vec0 = Vec(4, T).init(.{ self.data[1].v[0], self.data[0].v[0], self.data[0].v[0], self.data[0].v[0] });
                const Vec1 = Vec(4, T).init(.{ self.data[1].v[1], self.data[0].v[1], self.data[0].v[1], self.data[0].v[1] });
                const Vec2 = Vec(4, T).init(.{ self.data[1].v[2], self.data[0].v[2], self.data[0].v[2], self.data[0].v[2] });
                const Vec3 = Vec(4, T).init(.{ self.data[1].v[3], self.data[0].v[3], self.data[0].v[3], self.data[0].v[3] });

                const Inv0 = Vec1.mul(Fac0).sub(Vec2.mul(Fac1)).add(Vec3.mul(Fac2));
                const Inv1 = Vec0.mul(Fac0).sub(Vec2.mul(Fac3)).add(Vec3.mul(Fac4));
                const Inv2 = Vec0.mul(Fac1).sub(Vec1.mul(Fac3)).add(Vec3.mul(Fac5));
                const Inv3 = Vec0.mul(Fac2).sub(Vec1.mul(Fac4)).add(Vec2.mul(Fac5));

                const SignA = Vec(4, T).init(.{ @as(T, 1), @as(T, -1), @as(T, 1), @as(T, -1) });
                const SignB = Vec(4, T).init(.{ @as(T, -1), @as(T, 1), @as(T, -1), @as(T, 1) });
                const Inverse = init(.{
                    Inv0.mul(SignA),
                    Inv1.mul(SignB),
                    Inv2.mul(SignA),
                    Inv3.mul(SignB),
                });

                const Row0 = Vec(4, T).init(.{ Inverse.data[0].v[0], Inverse.data[1].v[0], Inverse.data[2].v[0], Inverse.data[3].v[0] });
                const Dot0 = self.data[0].mul(Row0);
                const Dot1 = (Dot0.v[0] + Dot0.v[1]) + (Dot0.v[2] + Dot0.v[3]);

                const OneOverDeterminant = @as(T, 1) / Dot1;
                return Inverse.mulScalar(OneOverDeterminant);
            } else {
                @compileError("inverse only supported for 2x2, 3x3 and 4x4 matrices");
            }
        }

        // ---- transforms (4x4 only) ----

        /// GLM `translate(m, v)`.
        pub fn translate(self: Self, v: Vec(3, T)) Self {
            if (comptime C != 4 or R != 4) @compileError("translate requires a 4x4 matrix");
            var res = self;
            res.data[3] = self.data[0].mul(v.v[0])
                .add(self.data[1].mul(v.v[1]))
                .add(self.data[2].mul(v.v[2]))
                .add(self.data[3]);
            return res;
        }

        /// GLM `scale(m, v)`.
        pub fn scale(self: Self, v: Vec(3, T)) Self {
            if (comptime C != 4 or R != 4) @compileError("scale requires a 4x4 matrix");
            var res: Self = undefined;
            res.data[0] = self.data[0].mul(v.v[0]);
            res.data[1] = self.data[1].mul(v.v[1]);
            res.data[2] = self.data[2].mul(v.v[2]);
            res.data[3] = self.data[3];
            return res;
        }

        /// GLM `rotate(m, angle, axis)`.
        pub fn rotate(self: Self, angle: T, axis: Vec(3, T)) Self {
            if (comptime C != 4 or R != 4) @compileError("rotate requires a 4x4 matrix");
            const c = scalar.cos(angle);
            const s = scalar.sin(angle);
            const axis_norm = axis.normalize();
            const temp = axis_norm.mul(@as(T, 1) - c);
            var rot = Self.zero();
            rot.data[0].v[0] = c + temp.v[0] * axis_norm.v[0];
            rot.data[0].v[1] = temp.v[0] * axis_norm.v[1] + s * axis_norm.v[2];
            rot.data[0].v[2] = temp.v[0] * axis_norm.v[2] - s * axis_norm.v[1];
            rot.data[1].v[0] = temp.v[1] * axis_norm.v[0] - s * axis_norm.v[2];
            rot.data[1].v[1] = c + temp.v[1] * axis_norm.v[1];
            rot.data[1].v[2] = temp.v[1] * axis_norm.v[2] + s * axis_norm.v[0];
            rot.data[2].v[0] = temp.v[2] * axis_norm.v[0] + s * axis_norm.v[1];
            rot.data[2].v[1] = temp.v[2] * axis_norm.v[1] - s * axis_norm.v[0];
            rot.data[2].v[2] = c + temp.v[2] * axis_norm.v[2];
            var res: Self = undefined;
            res.data[0] = self.data[0].mul(rot.data[0].v[0])
                .add(self.data[1].mul(rot.data[0].v[1]))
                .add(self.data[2].mul(rot.data[0].v[2]));
            res.data[1] = self.data[0].mul(rot.data[1].v[0])
                .add(self.data[1].mul(rot.data[1].v[1]))
                .add(self.data[2].mul(rot.data[1].v[2]));
            res.data[2] = self.data[0].mul(rot.data[2].v[0])
                .add(self.data[1].mul(rot.data[2].v[1]))
                .add(self.data[2].mul(rot.data[2].v[2]));
            res.data[3] = self.data[3];
            return res;
        }

        /// GLM `mat -> mat4` conversion (reproduces GLM's historical padding).
        pub fn toMat4(self: Self) Mat(4, 4, T) {
            if (comptime C == 4 and R == 4) return self;
            const c0 = self.data[0];
            const c1 = self.data[1];
            if (comptime C >= 3 and R >= 3) {
                // 3x3 / 3x4 / 4x3
                if (comptime R == 4) {
                    return Mat(4, 4, T).init(.{ c0, c1, self.data[2], Vec(4, T).init(.{ 0, 0, 0, 1 }) });
                } else if (comptime R == 3) {
                    return Mat(4, 4, T).init(.{
                        Vec(4, T).init(.{ c0.v[0], c0.v[1], c0.v[2], 0 }),
                        Vec(4, T).init(.{ c1.v[0], c1.v[1], c1.v[2], 0 }),
                        Vec(4, T).init(.{ self.data[2].v[0], self.data[2].v[1], self.data[2].v[2], 0 }),
                        Vec(4, T).init(.{ 0, 0, 0, 1 }),
                    });
                } else {
                    @compileError("toMat4: unsupported shape");
                }
            } else if (comptime C == 3 and R == 2) {
                return Mat(4, 4, T).init(.{
                    Vec(4, T).init(.{ c0.v[0], c0.v[1], 0, 0 }),
                    Vec(4, T).init(.{ c1.v[0], c1.v[1], 0, 0 }),
                    Vec(4, T).init(.{ self.data[2].v[0], self.data[2].v[1], 1, 0 }),
                    Vec(4, T).init(.{ 0, 0, 0, 1 }),
                });
            } else if (comptime C == 4 and R == 2) {
                return Mat(4, 4, T).init(.{
                    Vec(4, T).init(.{ c0.v[0], c0.v[1], 0, 0 }),
                    Vec(4, T).init(.{ c1.v[0], c1.v[1], 0, 0 }),
                    Vec(4, T).init(.{ 0, 0, 1, 0 }),
                    Vec(4, T).init(.{ 0, 0, 0, 1 }),
                });
            } else if (comptime C == 2) {
                const pad2 = Vec(4, T).init(.{ 0, 0, 1, 0 });
                const pad3 = Vec(4, T).init(.{ 0, 0, 0, 1 });
                if (comptime R == 4) {
                    return Mat(4, 4, T).init(.{ c0, c1, pad2, pad3 });
                } else if (comptime R == 3) {
                    return Mat(4, 4, T).init(.{
                        Vec(4, T).init(.{ c0.v[0], c0.v[1], c0.v[2], 0 }),
                        Vec(4, T).init(.{ c1.v[0], c1.v[1], c1.v[2], 0 }),
                        pad2,
                        pad3,
                    });
                } else {
                    return Mat(4, 4, T).init(.{
                        Vec(4, T).init(.{ c0.v[0], c0.v[1], 0, 0 }),
                        Vec(4, T).init(.{ c1.v[0], c1.v[1], 0, 0 }),
                        pad2,
                        pad3,
                    });
                }
            } else {
                @compileError("toMat4: unsupported shape");
            }
        }

        // ---- printing ----

        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.writeByte('{');
            for (0..C) |c| {
                for (0..R) |r| {
                    try writer.print("{d},", .{self.data[c].v[r]});
                }
            }
            try writer.writeByte('}');
        }

        // ---- ext/matrix_common ----

        /// Per-component abs (GLM `abs(mat)`).
        pub inline fn abs(self: Self) Self {
            var res: Self = undefined;
            inline for (0..C) |c| res.data[c] = self.data[c].abs();
            return res;
        }

        /// GLM `mix(x, y, a)` with a scalar or matrix interpolation factor.
        /// For a matrix factor GLM uses elementwise `1 - a` (the 1 is a scalar).
        pub fn mix(self: Self, rhs: Self, a: anytype) Self {
            const AT = @TypeOf(a);
            if (comptime isMat(AT)) {
                const ones_minus = ones().sub(a);
                return self.matrixCompMult(ones_minus).add(rhs.matrixCompMult(a));
            }
            const af: T = scalar.cast(T, a);
            var res: Self = undefined;
            inline for (0..C) |c| res.data[c] = self.data[c].mul(@as(T, 1) - af).add(rhs.data[c].mul(af));
            return res;
        }

        // ---- ext/matrix_relational ----

        fn colCmp(self: Self, rhs: Self, comptime need_all: bool, arg: anytype, comptime ulp: bool) Vec(C, bool) {
            const AT = @TypeOf(arg);
            const av = comptime vec.isVec(AT);
            var res: @Vector(C, bool) = undefined;
            inline for (0..C) |c| {
                const a = self.data[c];
                const b = rhs.data[c];
                const bit = if (comptime need_all) blk: {
                    if (comptime ulp) {
                        const e = a.equalULP(b, if (av) arg.v[c] else arg);
                        break :blk e.all();
                    } else {
                        const e = a.equalEps(b, if (av) arg.v[c] else arg);
                        break :blk e.all();
                    }
                } else blk: {
                    if (comptime ulp) {
                        const e = a.notEqualULP(b, if (av) arg.v[c] else arg);
                        break :blk e.any();
                    } else {
                        const e = a.notEqualEps(b, if (av) arg.v[c] else arg);
                        break :blk e.any();
                    }
                };
                res[c] = bit;
            }
            return .{ .v = res };
        }

        /// GLM `equal(mat, mat)` — per-column all-of-component equality.
        pub fn equal(self: Self, rhs: Self) Vec(C, bool) {
            var res: @Vector(C, bool) = undefined;
            inline for (0..C) |c| res[c] = self.data[c].equal(rhs.data[c]).all();
            return .{ .v = res };
        }

        /// GLM `notEqual(mat, mat)`.
        pub fn notEqual(self: Self, rhs: Self) Vec(C, bool) {
            var res: @Vector(C, bool) = undefined;
            inline for (0..C) |c| res[c] = self.data[c].notEqual(rhs.data[c]).any();
            return .{ .v = res };
        }

        /// GLM `equal(mat, mat, Epsilon)` — epsilon scalar or per-column vector.
        pub fn equalEps(self: Self, rhs: Self, eps: anytype) Vec(C, bool) {
            return self.colCmp(rhs, true, eps, false);
        }

        pub fn notEqualEps(self: Self, rhs: Self, eps: anytype) Vec(C, bool) {
            return self.colCmp(rhs, false, eps, false);
        }

        /// GLM `equal(mat, mat, MaxULPs)`.
        pub fn equalULP(self: Self, rhs: Self, max_ulps: anytype) Vec(C, bool) {
            return self.colCmp(rhs, true, max_ulps, true);
        }

        pub fn notEqualULP(self: Self, rhs: Self, max_ulps: anytype) Vec(C, bool) {
            return self.colCmp(rhs, false, max_ulps, true);
        }
    };
}

fn comptimePrint(comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.comptimePrint(fmt, args);
}

// ---- free functions (GLM ext/matrix_transform, ext/matrix_clip_space) ----

const mat4 = Mat(4, 4, f32);

/// GLM `lookAt` — default right-handed, NDC -1..1.
pub fn lookAt(eye: Vec(3, f32), center: Vec(3, f32), up: Vec(3, f32)) mat4 {
    const f = center.sub(eye).normalize();
    const s = f.cross(up).normalize();
    const u = s.cross(f);
    var res = mat4.identity();
    res.data[0].v[0] = s.v[0];
    res.data[1].v[0] = s.v[1];
    res.data[2].v[0] = s.v[2];
    res.data[0].v[1] = u.v[0];
    res.data[1].v[1] = u.v[1];
    res.data[2].v[1] = u.v[2];
    res.data[0].v[2] = -f.v[0];
    res.data[1].v[2] = -f.v[1];
    res.data[2].v[2] = -f.v[2];
    res.data[3].v[0] = -s.dot(eye);
    res.data[3].v[1] = -u.dot(eye);
    res.data[3].v[2] = f.dot(eye);
    return res;
}

/// GLM `lookAtLH` — left-handed, NDC -1..1.
pub fn lookAtLH(eye: Vec(3, f32), center: Vec(3, f32), up: Vec(3, f32)) mat4 {
    const f = center.sub(eye).normalize();
    const s = up.cross(f).normalize();
    const u = f.cross(s);
    var res = mat4.identity();
    res.data[0].v[0] = s.v[0];
    res.data[1].v[0] = s.v[1];
    res.data[2].v[0] = s.v[2];
    res.data[0].v[1] = u.v[0];
    res.data[1].v[1] = u.v[1];
    res.data[2].v[1] = u.v[2];
    res.data[0].v[2] = f.v[0];
    res.data[1].v[2] = f.v[1];
    res.data[2].v[2] = f.v[2];
    res.data[3].v[0] = -s.dot(eye);
    res.data[3].v[1] = -u.dot(eye);
    res.data[3].v[2] = -f.dot(eye);
    return res;
}

/// GLM `perspective` — right-handed, NDC -1..1 (default clip control).
pub fn perspective(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    const tan_half_fovy = scalar.tan(fovy / 2);
    var res = mat4.zero();
    res.data[0].v[0] = 1 / (aspect * tan_half_fovy);
    res.data[1].v[1] = 1 / tan_half_fovy;
    res.data[2].v[2] = -(zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveRH_ZO` — right-handed, NDC 0..1.
pub fn perspectiveRH_ZO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    const tan_half_fovy = scalar.tan(fovy / 2);
    var res = mat4.zero();
    res.data[0].v[0] = 1 / (aspect * tan_half_fovy);
    res.data[1].v[1] = 1 / tan_half_fovy;
    res.data[2].v[2] = zFar / (zNear - zFar);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveLH_ZO` — left-handed, NDC 0..1.
pub fn perspectiveLH_ZO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    const tan_half_fovy = scalar.tan(fovy / 2);
    var res = mat4.zero();
    res.data[0].v[0] = 1 / (aspect * tan_half_fovy);
    res.data[1].v[1] = 1 / tan_half_fovy;
    res.data[2].v[2] = zFar / (zFar - zNear);
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -(zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `ortho` — right-handed, NDC -1..1 (default clip control).
pub fn ortho(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = -2 / (zFar - zNear);
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    res.data[3].v[2] = -(zFar + zNear) / (zFar - zNear);
    res.data[3].v[3] = 1;
    return res;
}

/// GLM `orthoRH_ZO` — right-handed, NDC 0..1.
pub fn orthoRH_ZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = -1 / (zFar - zNear);
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    res.data[3].v[2] = -zNear / (zFar - zNear);
    res.data[3].v[3] = 1;
    return res;
}

/// GLM `frustum` — right-handed, NDC -1..1 (default clip control).
pub fn frustum(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[0] = (right + left) / (right - left);
    res.data[2].v[1] = (top + bottom) / (top - bottom);
    res.data[2].v[2] = -(zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `lookAtRH` — same math as `lookAt` (default clip control is RH_NO).
pub fn lookAtRH(eye: Vec(3, f32), center: Vec(3, f32), up: Vec(3, f32)) mat4 {
    return lookAt(eye, center, up);
}

/// GLM `perspectiveRH_NO` — right-handed, NDC -1..1.
pub fn perspectiveRH_NO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspective(fovy, aspect, zNear, zFar);
}

/// GLM `perspectiveLH_NO` — left-handed, NDC -1..1.
pub fn perspectiveLH_NO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    const tan_half_fovy = scalar.tan(fovy / 2);
    var res = mat4.zero();
    res.data[0].v[0] = 1 / (aspect * tan_half_fovy);
    res.data[1].v[1] = 1 / tan_half_fovy;
    res.data[2].v[2] = (zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveZO` (default: right-handed, ZO).
pub fn perspectiveZO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveRH_ZO(fovy, aspect, zNear, zFar);
}

/// GLM `perspectiveNO` (default: right-handed, NO).
pub fn perspectiveNO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspective(fovy, aspect, zNear, zFar);
}

/// GLM `perspectiveLH` (default clip control: NO).
pub fn perspectiveLH(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveLH_NO(fovy, aspect, zNear, zFar);
}

/// GLM `perspectiveRH` (default clip control: NO).
pub fn perspectiveRH(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspective(fovy, aspect, zNear, zFar);
}

/// GLM `perspectiveFovRH_ZO` — fov in radians for a given width/height.
pub fn perspectiveFovRH_ZO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    const h = scalar.cos(0.5 * fov) / scalar.sin(0.5 * fov);
    const w = h * height / width;
    var res = mat4.zero();
    res.data[0].v[0] = w;
    res.data[1].v[1] = h;
    res.data[2].v[2] = zFar / (zNear - zFar);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveFovRH_NO`.
pub fn perspectiveFovRH_NO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    const h = scalar.cos(0.5 * fov) / scalar.sin(0.5 * fov);
    const w = h * height / width;
    var res = mat4.zero();
    res.data[0].v[0] = w;
    res.data[1].v[1] = h;
    res.data[2].v[2] = -(zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveFovLH_ZO`.
pub fn perspectiveFovLH_ZO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    const h = scalar.cos(0.5 * fov) / scalar.sin(0.5 * fov);
    const w = h * height / width;
    var res = mat4.zero();
    res.data[0].v[0] = w;
    res.data[1].v[1] = h;
    res.data[2].v[2] = zFar / (zFar - zNear);
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -(zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveFovLH_NO`.
pub fn perspectiveFovLH_NO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    const h = scalar.cos(0.5 * fov) / scalar.sin(0.5 * fov);
    const w = h * height / width;
    var res = mat4.zero();
    res.data[0].v[0] = w;
    res.data[1].v[1] = h;
    res.data[2].v[2] = (zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `perspectiveFov` (default: right-handed, NO).
pub fn perspectiveFov(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
}

/// GLM `perspectiveFovZO` (default: right-handed).
pub fn perspectiveFovZO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_ZO(fov, width, height, zNear, zFar);
}

/// GLM `perspectiveFovNO` (default: right-handed).
pub fn perspectiveFovNO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
}

/// GLM `perspectiveFovLH` (default clip control: NO).
pub fn perspectiveFovLH(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovLH_NO(fov, width, height, zNear, zFar);
}

/// GLM `perspectiveFovRH` (default clip control: NO).
pub fn perspectiveFovRH(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
}

/// GLM `infinitePerspectiveRH_NO` — infinite far plane, NDC -1..1.
pub fn infinitePerspectiveRH_NO(fovy: f32, aspect: f32, zNear: f32) mat4 {
    const range = scalar.tan(fovy / 2) * zNear;
    const left = -range * aspect;
    const right = range * aspect;
    const bottom = -range;
    const top = range;
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[2] = -1;
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -2 * zNear;
    return res;
}

/// GLM `infinitePerspectiveRH_ZO`.
pub fn infinitePerspectiveRH_ZO(fovy: f32, aspect: f32, zNear: f32) mat4 {
    const range = scalar.tan(fovy / 2) * zNear;
    const left = -range * aspect;
    const right = range * aspect;
    const bottom = -range;
    const top = range;
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[2] = -1;
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -zNear;
    return res;
}

/// GLM `infinitePerspectiveLH_NO`.
pub fn infinitePerspectiveLH_NO(fovy: f32, aspect: f32, zNear: f32) mat4 {
    const range = scalar.tan(fovy / 2) * zNear;
    const left = -range * aspect;
    const right = range * aspect;
    const bottom = -range;
    const top = range;
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[2] = 1;
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -2 * zNear;
    return res;
}

/// GLM `infinitePerspectiveLH_ZO`.
pub fn infinitePerspectiveLH_ZO(fovy: f32, aspect: f32, zNear: f32) mat4 {
    const range = scalar.tan(fovy / 2) * zNear;
    const left = -range * aspect;
    const right = range * aspect;
    const bottom = -range;
    const top = range;
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[2] = 1;
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -zNear;
    return res;
}

/// GLM `infinitePerspective` (default: right-handed, NO).
pub fn infinitePerspective(fovy: f32, aspect: f32, zNear: f32) mat4 {
    return infinitePerspectiveRH_NO(fovy, aspect, zNear);
}

/// GLM `tweakedInfinitePerspective(fovy, aspect, zNear, ep)` — Lengyel's tweak.
pub fn tweakedInfinitePerspective(fovy: f32, aspect: f32, zNear: f32, ep: f32) mat4 {
    const range = scalar.tan(fovy / 2) * zNear;
    const left = -range * aspect;
    const right = range * aspect;
    const bottom = -range;
    const top = range;
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[2] = ep - 1;
    res.data[2].v[3] = -1;
    res.data[3].v[2] = (ep - 2) * zNear;
    return res;
}

/// GLM `tweakedInfinitePerspective(fovy, aspect, zNear)` — ep = machine epsilon.
pub fn tweakedInfinitePerspectiveDefault(fovy: f32, aspect: f32, zNear: f32) mat4 {
    return tweakedInfinitePerspective(fovy, aspect, zNear, std.math.floatEps(f32));
}

/// GLM `ortho(left, right, bottom, top)` — 2D, no depth range.
pub fn ortho2D(left: f32, right: f32, bottom: f32, top: f32) mat4 {
    var res = mat4.identity();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = -1;
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    return res;
}

/// GLM `orthoLH_ZO`.
pub fn orthoLH_ZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.identity();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = 1 / (zFar - zNear);
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    res.data[3].v[2] = -zNear / (zFar - zNear);
    return res;
}

/// GLM `orthoLH_NO`.
pub fn orthoLH_NO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.identity();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = 2 / (zFar - zNear);
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    res.data[3].v[2] = -(zFar + zNear) / (zFar - zNear);
    return res;
}

/// GLM `orthoRH_NO`.
pub fn orthoRH_NO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.identity();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = -2 / (zFar - zNear);
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    res.data[3].v[2] = -(zFar + zNear) / (zFar - zNear);
    return res;
}

/// GLM `orthoZO` (default: right-handed).
pub fn orthoZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoRH_ZO(left, right, bottom, top, zNear, zFar);
}

/// GLM `orthoNO` (default: right-handed).
pub fn orthoNO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoRH_NO(left, right, bottom, top, zNear, zFar);
}

/// GLM `orthoLH` (default clip control: NO).
pub fn orthoLH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoLH_NO(left, right, bottom, top, zNear, zFar);
}

/// GLM `orthoRH` (default clip control: NO).
pub fn orthoRH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoRH_NO(left, right, bottom, top, zNear, zFar);
}

/// GLM `frustumLH_ZO`.
pub fn frustumLH_ZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[0] = -(right + left) / (right - left);
    res.data[2].v[1] = -(top + bottom) / (top - bottom);
    res.data[2].v[2] = zFar / (zFar - zNear);
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -(zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `frustumLH_NO`.
pub fn frustumLH_NO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[0] = -(right + left) / (right - left);
    res.data[2].v[1] = -(top + bottom) / (top - bottom);
    res.data[2].v[2] = (zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = 1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `frustumRH_ZO`.
pub fn frustumRH_ZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[0] = (right + left) / (right - left);
    res.data[2].v[1] = (top + bottom) / (top - bottom);
    res.data[2].v[2] = zFar / (zNear - zFar);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `frustumRH_NO`.
pub fn frustumRH_NO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[0] = (right + left) / (right - left);
    res.data[2].v[1] = (top + bottom) / (top - bottom);
    res.data[2].v[2] = -(zFar + zNear) / (zFar - zNear);
    res.data[2].v[3] = -1;
    res.data[3].v[2] = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

/// GLM `frustumZO` (default: right-handed).
pub fn frustumZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumRH_ZO(left, right, bottom, top, zNear, zFar);
}

/// GLM `frustumNO` (default: right-handed).
pub fn frustumNO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumRH_NO(left, right, bottom, top, zNear, zFar);
}

/// GLM `frustumLH` (default clip control: NO).
pub fn frustumLH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumLH_NO(left, right, bottom, top, zNear, zFar);
}

/// GLM `frustumRH` (default clip control: NO).
pub fn frustumRH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumRH_NO(left, right, bottom, top, zNear, zFar);
}

// ---- GLM ext/matrix_projection ----

/// GLM `projectZO(obj, model, proj, viewport)` — NDC 0..1.
pub fn projectZO(obj: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    var tmp = proj.mulVec(model.mulVec(Vec(4, f32).init(.{ obj.v[0], obj.v[1], obj.v[2], 1 })));
    tmp = tmp.div(tmp.v[3]);
    tmp.v[0] = tmp.v[0] * 0.5 + 0.5;
    tmp.v[1] = tmp.v[1] * 0.5 + 0.5;
    tmp.v[0] = tmp.v[0] * viewport.v[2] + viewport.v[0];
    tmp.v[1] = tmp.v[1] * viewport.v[3] + viewport.v[1];
    return Vec(3, f32).init(.{ tmp.v[0], tmp.v[1], tmp.v[2] });
}

/// GLM `projectNO(obj, model, proj, viewport)` — NDC -1..1.
pub fn projectNO(obj: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    var tmp = proj.mulVec(model.mulVec(Vec(4, f32).init(.{ obj.v[0], obj.v[1], obj.v[2], 1 })));
    tmp = tmp.div(tmp.v[3]);
    tmp = tmp.mul(0.5).add(0.5);
    tmp.v[0] = tmp.v[0] * viewport.v[2] + viewport.v[0];
    tmp.v[1] = tmp.v[1] * viewport.v[3] + viewport.v[1];
    return Vec(3, f32).init(.{ tmp.v[0], tmp.v[1], tmp.v[2] });
}

/// GLM `project` (default clip control: NO).
pub fn project(obj: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    return projectNO(obj, model, proj, viewport);
}

/// GLM `unProjectZO(win, model, proj, viewport)`.
pub fn unProjectZO(win: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    const inv = proj.mul(model).inverse();
    var tmp: Vec(4, f32) = undefined;
    tmp.v[0] = win.v[0];
    tmp.v[1] = win.v[1];
    tmp.v[2] = win.v[2];
    tmp.v[3] = 1;
    tmp.v[0] = (tmp.v[0] - viewport.v[0]) / viewport.v[2];
    tmp.v[1] = (tmp.v[1] - viewport.v[1]) / viewport.v[3];
    tmp.v[0] = tmp.v[0] * 2 - 1;
    tmp.v[1] = tmp.v[1] * 2 - 1;
    var obj = inv.mulVec(tmp);
    obj = obj.div(obj.v[3]);
    return Vec(3, f32).init(.{ obj.v[0], obj.v[1], obj.v[2] });
}

/// GLM `unProjectNO(win, model, proj, viewport)`.
pub fn unProjectNO(win: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    const inv = proj.mul(model).inverse();
    var tmp: Vec(4, f32) = undefined;
    tmp.v[0] = win.v[0];
    tmp.v[1] = win.v[1];
    tmp.v[2] = win.v[2];
    tmp.v[3] = 1;
    tmp.v[0] = (tmp.v[0] - viewport.v[0]) / viewport.v[2];
    tmp.v[1] = (tmp.v[1] - viewport.v[1]) / viewport.v[3];
    tmp = tmp.mul(2).sub(1);
    var obj = inv.mulVec(tmp);
    obj = obj.div(obj.v[3]);
    return Vec(3, f32).init(.{ obj.v[0], obj.v[1], obj.v[2] });
}

/// GLM `unProject` (default clip control: NO).
pub fn unProject(win: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    return unProjectNO(win, model, proj, viewport);
}

/// GLM `pickMatrix(center, delta, viewport)`.
pub fn pickMatrix(center: Vec(2, f32), delta: Vec(2, f32), viewport: Vec(4, f32)) mat4 {
    var res = mat4.identity();
    if (!(delta.v[0] > 0 and delta.v[1] > 0)) return res;
    const temp = Vec(3, f32).init(.{
        (viewport.v[2] - 2 * (center.v[0] - viewport.v[0])) / delta.v[0],
        (viewport.v[3] - 2 * (center.v[1] - viewport.v[1])) / delta.v[1],
        0,
    });
    res = res.translate(temp);
    return res.scale(Vec(3, f32).init(.{ viewport.v[2] / delta.v[0], viewport.v[3] / delta.v[1], 1 }));
}

// ---- GLM gtc/matrix_inverse ----

/// GLM `affineInverse` for 3x3 — 2x2 inverse of the upper-left, no translation.
pub fn affineInverse3(m: Mat(3, 3, f32)) Mat(3, 3, f32) {
    const inv22 = Mat(2, 2, f32).init(.{
        Vec(2, f32).init(.{ m.data[0].v[0], m.data[0].v[1] }),
        Vec(2, f32).init(.{ m.data[1].v[0], m.data[1].v[1] }),
    }).inverse();
    const t = Vec(2, f32).init(.{ m.data[2].v[0], m.data[2].v[1] });
    const neg = inv22.mulVec(t).neg();
    return Mat(3, 3, f32).init(.{
        Vec(3, f32).init(.{ inv22.data[0].v[0], inv22.data[0].v[1], 0 }),
        Vec(3, f32).init(.{ inv22.data[1].v[0], inv22.data[1].v[1], 0 }),
        Vec(3, f32).init(.{ neg.v[0], neg.v[1], 1 }),
    });
}

/// GLM `affineInverse` for 4x4.
pub fn affineInverse(m: mat4) mat4 {
    const inv33 = Mat(3, 3, f32).init(.{
        Vec(3, f32).init(.{ m.data[0].v[0], m.data[0].v[1], m.data[0].v[2] }),
        Vec(3, f32).init(.{ m.data[1].v[0], m.data[1].v[1], m.data[1].v[2] }),
        Vec(3, f32).init(.{ m.data[2].v[0], m.data[2].v[1], m.data[2].v[2] }),
    }).inverse();
    const t = Vec(3, f32).init(.{ m.data[3].v[0], m.data[3].v[1], m.data[3].v[2] });
    const neg = inv33.mulVec(t).neg();
    return mat4.init(.{
        Vec(4, f32).init(.{ inv33.data[0].v[0], inv33.data[0].v[1], inv33.data[0].v[2], 0 }),
        Vec(4, f32).init(.{ inv33.data[1].v[0], inv33.data[1].v[1], inv33.data[1].v[2], 0 }),
        Vec(4, f32).init(.{ inv33.data[2].v[0], inv33.data[2].v[1], inv33.data[2].v[2], 0 }),
        Vec(4, f32).init(.{ neg.v[0], neg.v[1], neg.v[2], 1 }),
    });
}

/// GLM `inverseTranspose` 2x2 — note GLM 1.1.0 returns the inverse here (not transposed).
pub fn inverseTranspose2(m: Mat(2, 2, f32)) Mat(2, 2, f32) {
    const d = m.determinant();
    return Mat(2, 2, f32).init(.{
        Vec(2, f32).init(.{ m.data[1].v[1] / d, -m.data[0].v[1] / d }),
        Vec(2, f32).init(.{ -m.data[1].v[0] / d, m.data[0].v[0] / d }),
    });
}

/// GLM `inverseTranspose` 3x3.
pub fn inverseTranspose3(m: Mat(3, 3, f32)) Mat(3, 3, f32) {
    const determinant =
        m.data[0].v[0] * (m.data[1].v[1] * m.data[2].v[2] - m.data[1].v[2] * m.data[2].v[1]) -
        m.data[0].v[1] * (m.data[1].v[0] * m.data[2].v[2] - m.data[1].v[2] * m.data[2].v[0]) +
        m.data[0].v[2] * (m.data[1].v[0] * m.data[2].v[1] - m.data[1].v[1] * m.data[2].v[0]);
    var inv: Mat(3, 3, f32) = undefined;
    inv.data[0].v[0] = (m.data[1].v[1] * m.data[2].v[2] - m.data[2].v[1] * m.data[1].v[2]);
    inv.data[0].v[1] = -(m.data[1].v[0] * m.data[2].v[2] - m.data[2].v[0] * m.data[1].v[2]);
    inv.data[0].v[2] = (m.data[1].v[0] * m.data[2].v[1] - m.data[2].v[0] * m.data[1].v[1]);
    inv.data[1].v[0] = -(m.data[0].v[1] * m.data[2].v[2] - m.data[2].v[1] * m.data[0].v[2]);
    inv.data[1].v[1] = (m.data[0].v[0] * m.data[2].v[2] - m.data[2].v[0] * m.data[0].v[2]);
    inv.data[1].v[2] = -(m.data[0].v[0] * m.data[2].v[1] - m.data[2].v[0] * m.data[0].v[1]);
    inv.data[2].v[0] = (m.data[0].v[1] * m.data[1].v[2] - m.data[1].v[1] * m.data[0].v[2]);
    inv.data[2].v[1] = -(m.data[0].v[0] * m.data[1].v[2] - m.data[1].v[0] * m.data[0].v[2]);
    inv.data[2].v[2] = (m.data[0].v[0] * m.data[1].v[1] - m.data[1].v[0] * m.data[0].v[1]);
    var res = inv;
    inline for (0..3) |c| {
        inline for (0..3) |r| res.data[c].v[r] /= determinant;
    }
    return res;
}

/// GLM `inverseTranspose` 4x4.
pub fn inverseTranspose(m: mat4) mat4 {
    const sf00 = m.data[2].v[2] * m.data[3].v[3] - m.data[3].v[2] * m.data[2].v[3];
    const sf01 = m.data[2].v[1] * m.data[3].v[3] - m.data[3].v[1] * m.data[2].v[3];
    const sf02 = m.data[2].v[1] * m.data[3].v[2] - m.data[3].v[1] * m.data[2].v[2];
    const sf03 = m.data[2].v[0] * m.data[3].v[3] - m.data[3].v[0] * m.data[2].v[3];
    const sf04 = m.data[2].v[0] * m.data[3].v[2] - m.data[3].v[0] * m.data[2].v[2];
    const sf05 = m.data[2].v[0] * m.data[3].v[1] - m.data[3].v[0] * m.data[2].v[1];
    const sf06 = m.data[1].v[2] * m.data[3].v[3] - m.data[3].v[2] * m.data[1].v[3];
    const sf07 = m.data[1].v[1] * m.data[3].v[3] - m.data[3].v[1] * m.data[1].v[3];
    const sf08 = m.data[1].v[1] * m.data[3].v[2] - m.data[3].v[1] * m.data[1].v[2];
    const sf09 = m.data[1].v[0] * m.data[3].v[3] - m.data[3].v[0] * m.data[1].v[3];
    const sf10 = m.data[1].v[0] * m.data[3].v[2] - m.data[3].v[0] * m.data[1].v[2];
    const sf11 = m.data[1].v[0] * m.data[3].v[1] - m.data[3].v[0] * m.data[1].v[1];
    const sf12 = m.data[1].v[2] * m.data[2].v[3] - m.data[2].v[2] * m.data[1].v[3];
    const sf13 = m.data[1].v[1] * m.data[2].v[3] - m.data[2].v[1] * m.data[1].v[3];
    const sf14 = m.data[1].v[1] * m.data[2].v[2] - m.data[2].v[1] * m.data[1].v[2];
    const sf15 = m.data[1].v[0] * m.data[2].v[3] - m.data[2].v[0] * m.data[1].v[3];
    const sf16 = m.data[1].v[0] * m.data[2].v[2] - m.data[2].v[0] * m.data[1].v[2];
    const sf17 = m.data[1].v[0] * m.data[2].v[1] - m.data[2].v[0] * m.data[1].v[1];

    var inv: mat4 = undefined;
    inv.data[0].v[0] = m.data[1].v[1] * sf00 - m.data[1].v[2] * sf01 + m.data[1].v[3] * sf02;
    inv.data[0].v[1] = -(m.data[1].v[0] * sf00 - m.data[1].v[2] * sf03 + m.data[1].v[3] * sf04);
    inv.data[0].v[2] = m.data[1].v[0] * sf01 - m.data[1].v[1] * sf03 + m.data[1].v[3] * sf05;
    inv.data[0].v[3] = -(m.data[1].v[0] * sf02 - m.data[1].v[1] * sf04 + m.data[1].v[2] * sf05);

    inv.data[1].v[0] = -(m.data[0].v[1] * sf00 - m.data[0].v[2] * sf01 + m.data[0].v[3] * sf02);
    inv.data[1].v[1] = m.data[0].v[0] * sf00 - m.data[0].v[2] * sf03 + m.data[0].v[3] * sf04;
    inv.data[1].v[2] = -(m.data[0].v[0] * sf01 - m.data[0].v[1] * sf03 + m.data[0].v[3] * sf05);
    inv.data[1].v[3] = m.data[0].v[0] * sf02 - m.data[0].v[1] * sf04 + m.data[0].v[2] * sf05;

    inv.data[2].v[0] = m.data[0].v[1] * sf06 - m.data[0].v[2] * sf07 + m.data[0].v[3] * sf08;
    inv.data[2].v[1] = -(m.data[0].v[0] * sf06 - m.data[0].v[2] * sf09 + m.data[0].v[3] * sf10);
    inv.data[2].v[2] = m.data[0].v[0] * sf07 - m.data[0].v[1] * sf09 + m.data[0].v[3] * sf11;
    inv.data[2].v[3] = -(m.data[0].v[0] * sf08 - m.data[0].v[1] * sf10 + m.data[0].v[2] * sf11);

    inv.data[3].v[0] = -(m.data[0].v[1] * sf12 - m.data[0].v[2] * sf13 + m.data[0].v[3] * sf14);
    inv.data[3].v[1] = m.data[0].v[0] * sf12 - m.data[0].v[2] * sf15 + m.data[0].v[3] * sf16;
    inv.data[3].v[2] = -(m.data[0].v[0] * sf13 - m.data[0].v[1] * sf15 + m.data[0].v[3] * sf17);
    inv.data[3].v[3] = m.data[0].v[0] * sf14 - m.data[0].v[1] * sf16 + m.data[0].v[2] * sf17;

    const determinant =
        m.data[0].v[0] * inv.data[0].v[0] +
        m.data[0].v[1] * inv.data[0].v[1] +
        m.data[0].v[2] * inv.data[0].v[2] +
        m.data[0].v[3] * inv.data[0].v[3];

    var res = inv;
    inline for (0..4) |c| {
        inline for (0..4) |r| res.data[c].v[r] /= determinant;
    }
    return res;
}
pub fn outerProduct(c: anytype, r: anytype) Mat(@TypeOf(r).len, @TypeOf(c).len, @TypeOf(c).value_type) {
    const T = @TypeOf(c).value_type;
    const CL = @TypeOf(c).len;
    const RL = @TypeOf(r).len;
    var res: Mat(RL, CL, T) = undefined;
    inline for (0..RL) |i| {
        inline for (0..CL) |j| res.data[i].v[j] = c.v[j] * r.v[i];
    }
    return res;
}
