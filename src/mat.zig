//! Matrix math — GLM-equivalent `mat<C, R, T>` implementation.
//! Column-major storage: `data[c]` is column c, `data[c][r]` is element at
//! (column c, row r), matching GLM's `m[c][r]` indexing.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const Vec = vec.Vec;

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

/// GLM `outerProduct(c, r)` — result has `r.len` columns and `c.len` rows.
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
