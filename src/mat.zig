//! Matrix math — GLM-compatible `mat<num_columns, num_rows, scalar_type>` (num_columns columns, num_rows rows).
//!
//! Column-major storage: `data[c]` is column c, `data[c][r]` is the element
//! at (column c, row r), mirroring GLM's `m[c][r]` indexing and GLSL's
//! column-major layout. Consequently `mul` treats the receiver as the
//! LEFT operand: `matrix.mulVec(v)` computes `matrix * v`, and `a.mul(b)` computes
//! `a * b` — the order matters, matrix products do not commute.
//!
//! Affine 4x4 helpers (`translate`/`scale`/`rotate`) mutate the receiver
//! like GLM's `glm::translate(m, ...)`, i.e. they append the transform to
//! an already built matrix instead of prescribing how to build one from
//! scratch; combine them left-to-right in the order you want applied.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const Vec = vec.Vec;

/// Returns `true` if `scalar_type` looks like a matrix produced by `Mat(num_columns, num_rows, scalar_type)`
/// (it declares `cols`/`rows` and carries a `data` field). Use to dispatch
/// between scalar/vector/matrix operands in generic code.
pub fn isMat(comptime candidate_type: type) bool {
    return @typeInfo(candidate_type) == .@"struct" and @hasDecl(candidate_type, "cols") and @hasDecl(candidate_type, "rows") and @hasField(candidate_type, "data");
}

/// Create a matrix type with `num_columns` columns and `num_rows` rows of scalar `scalar_type`
/// (float or int). The type name doubles as a namespace:
/// `mat4.identity()`, `Mat(3, 3, f32).zero()`, and the type exposes
/// `cols`/`rows`/`col_type`/`row_type` metadata for generic code.
pub fn Mat(comptime num_columns: usize, comptime num_rows: usize, comptime scalar_type: type) type {
    return struct {
        pub const Self = @This();
        pub const cols: comptime_int = num_columns;
        pub const rows: comptime_int = num_rows;
        pub const col_count: comptime_int = num_columns;
        pub const row_count: comptime_int = num_rows;
        pub const value_type: type = scalar_type;
        pub const col_type: type = Vec(num_rows, scalar_type);
        pub const row_type: type = Vec(num_columns, scalar_type);

        data: [num_columns]Vec(num_rows, scalar_type),

        // ---- constructors ----

        /// Matrix of all zeroes (GLM `mat(num_columns, num_rows, 0)`). The additive identity
        /// and the result of `matrix.sub(m)`; a projection built from scratch
        /// usually starts here before its few non-zero elements are placed.
        pub fn zero() Self {
            return .{ .data = [_]Vec(num_rows, scalar_type){Vec(num_rows, scalar_type).zero()} ** num_columns };
        }

        /// Square identity matrix (GLM `mat(v)` with v=1): ones on the
        /// diagonal, zeroes elsewhere. Multiplying by it changes nothing;
        /// use it as the starting point of `translate`/`scale`/`rotate`
        /// chains.
        pub fn identity() Self {
            if (comptime num_columns != num_rows) @compileError("identity requires a square matrix");
            return diag(@as(scalar_type, 1));
        }

        /// Matrix with every element equal to `v` (GLM 1.1 `mat(f)`
        /// elementwise variant). Useful to build per-element factors, e.g.
        /// GLM's matrix `mix` uses `ones() - a`.
        pub fn one(value: anytype) Self {
            return .{ .data = [_]Vec(num_rows, scalar_type){Vec(num_rows, scalar_type).fill(scalar.cast(scalar_type, value))} ** num_columns };
        }

        /// Matrix of all ones (GLM `mat(1)` fill style). Comes in handy as
        /// the "1" in element-wise expressions like `ones() - a`.
        pub fn ones() Self {
            return one(@as(scalar_type, 1));
        }

        /// Diagonal matrix with value `v` on the diagonal (GLM `mat(v)` for
        /// a scalar v): `diag(1)` is `identity()`, `diag(2)` scales by 2.
        pub fn diag(value: anytype) Self {
            if (comptime num_columns != num_rows) @compileError("diag requires a square matrix");
            var res = zero();
            inline for (0..num_columns) |i| res.data[i].v[i] = scalar.cast(scalar_type, value);
            return res;
        }

        /// Build a matrix from a tuple of columns, each a `Vec(num_rows, scalar_type)`
        /// (GLM's `mat(c0, c1, ...)` column constructor), or from a plain
        /// scalar which produces `diag(v)`. Use for literals:
        /// `mat4.init(.{ c0, c1, c2, c3 })` where `cN` are `vec4`s.
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
                if (comptime !(ET == Vec(num_rows, scalar_type)))
                    @compileError("Mat init: expected column of type " ++ @typeName(Vec(num_rows, scalar_type)));
                if (comptime n >= num_columns) @compileError("Mat init: too many columns");
                res.data[n] = e;
                n += 1;
            }
            if (comptime n != num_columns) @compileError("Mat init: expected " ++ comptimePrint("{d}", .{num_columns}) ++ " columns");
            return res;
        }

        // ---- accessors ----

        /// Read element at (column `c`, row `r`), 0-based, runtime indices.
        /// Prefer `col(i).v[r]` directly for hot paths — it compiles to a
        /// lane load without the extra call.
        pub inline fn get(self: Self, col_index: usize, row_index: usize) scalar_type {
            return self.data[col_index].v[row_index];
        }

        /// Write element at (column `c`, row `r`) of a mutable matrix.
        pub inline fn set(self: *Self, col_index: usize, row_index: usize, value: scalar_type) void {
            self.data[col_index].v[row_index] = value;
        }

        /// Column c as a vector (GLM `column(m, c)`); indexes 0-based.
        /// Reading a column is free — it is the native storage unit.
        pub inline fn col(self: Self, i: usize) Vec(num_rows, scalar_type) {
            return self.data[i];
        }

        /// Row r as a vector (GLM `row(m, r)`), gathered across columns.
        /// Costlier than `col`; use for dot products with row-major data.
        pub fn row(self: Self, i: usize) Vec(num_columns, scalar_type) {
            var res: Vec(num_columns, scalar_type) = undefined;
            inline for (0..num_columns) |c| res.v[c] = self.data[c].v[i];
            return res;
        }

        // ---- arithmetic ----

        /// Element-wise addition (GLM `mat + mat`). Note this is NOT a
        /// linear-algebra operation — added matrices must share the same
        /// shape, and the result is per-element sums (rarely meaningful
        /// for transforms; useful for blending/interpolating weights).
        pub fn add(self: Self, right_hand_side: Self) Self {
            var res: Self = undefined;
            inline for (0..num_columns) |c| res.data[c] = self.data[c].add(right_hand_side.data[c]);
            return res;
        }

        /// Element-wise subtraction (GLM `mat - mat`); inverse of `add`.
        pub fn sub(self: Self, right_hand_side: Self) Self {
            var res: Self = undefined;
            inline for (0..num_columns) |c| res.data[c] = self.data[c].sub(right_hand_side.data[c]);
            return res;
        }

        /// Scalar multiplication: every element × `s` (GLM `mat * scalar`).
        /// Use to scale a matrix's effect (e.g. dampen a correction step)
        /// or with `1/det` when inverting.
        pub fn mulScalar(self: Self, scalar_value: anytype) Self {
            var res: Self = undefined;
            inline for (0..num_columns) |c| res.data[c] = self.data[c].mul(scalar_value);
            return res;
        }

        /// Component-wise (Hadamard) product (GLM `matrixCompMult`):
        /// `res[c][r] = a[c][r] * b[c][r]`. Distinct from real matrix
        /// multiplication `mul`; GLM uses it internally for its `mix`.
        pub fn matrixCompMult(self: Self, right_hand_side: Self) Self {
            var res: Self = undefined;
            inline for (0..num_columns) |c| res.data[c] = self.data[c].mul(right_hand_side.data[c]);
            return res;
        }

        /// Matrix product `self * m` (GLM `mat * mat`): column-major
        /// multiplication. The result has `m`'s column count and `self`'s
        /// row count (so `Mat(3,2).mul(Mat(4,3))` is a 4×2 matrix — GLM's
        /// `mat<num_columns,num_rows>*mat<C2,num_columns>` shape rule). Use for composing transforms
        /// as `model.mul(view)` (view applies first). Order matters:
        /// `a.mul(b)` is NOT `b.mul(a)`.
        pub fn mul(self: Self, right_hand_side: anytype) Mat(@TypeOf(right_hand_side).cols, num_rows, scalar_type) {
            const right_hand_side_col_count = @TypeOf(right_hand_side).cols;
            var res: Mat(right_hand_side_col_count, num_rows, scalar_type) = undefined;
            inline for (0..right_hand_side_col_count) |j| res.data[j] = self.mulVec(right_hand_side.col(j));
            return res;
        }

        /// Matrix × vector: `self * v` (GLM `mat * vec`). Computed as a
        /// linear combination of `self`'s columns, which is why column-major
        /// storage exists. This is the "transform a point/direction" call:
        /// `viewProj.mulVec(pos4)`.
        pub fn mulVec(self: Self, vector: Vec(num_columns, scalar_type)) Vec(num_rows, scalar_type) {
            var res: Vec(num_rows, scalar_type) = undefined;
            inline for (0..num_rows) |r| {
                var acc: scalar_type = undefined;
                inline for (0..num_columns) |c| {
                    const term = self.data[c].v[r] * vector.v[c];
                    acc = if (c == 0) term else acc + term;
                }
                res.v[r] = acc;
            }
            return res;
        }

        /// Transpose (GLM `transpose`): swaps rows and columns, producing
        /// an num_rows×num_columns matrix. `matrix.transpose().transpose() == m`; used when
        /// converting between column- and row-major conventions and when
        /// inverting rotation matrices (the inverse of an orthogonal matrix
        /// is its transpose).
        pub fn transpose(self: Self) Mat(num_rows, num_columns, scalar_type) {
            var res: Mat(num_rows, num_columns, scalar_type) = undefined;
            inline for (0..num_rows) |r| {
                inline for (0..num_columns) |c| res.data[r].v[c] = self.data[c].v[r];
            }
            return res;
        }

        /// Determinant (GLM `determinant`; 2x2/3x3/4x4 only): the signed
        /// volume scaling factor of the linear map. Zero (or near-zero)
        /// means the matrix is singular — `inverse` will divide by it, so
        /// test `det != 0` before inverting.
        pub fn determinant(self: Self) scalar_type {
            if (comptime num_columns != num_rows) @compileError("determinant requires a square matrix");
            if (comptime num_rows == 2) {
                return self.data[0].v[0] * self.data[1].v[1] - self.data[1].v[0] * self.data[0].v[1];
            } else if (comptime num_rows == 3) {
                const t1 = self.data[1].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[1].v[2];
                const t2 = self.data[0].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[0].v[2];
                const t3 = self.data[0].v[1] * self.data[1].v[2] - self.data[1].v[1] * self.data[0].v[2];
                return self.data[0].v[0] * t1 - self.data[1].v[0] * t2 + self.data[2].v[0] * t3;
            } else if (comptime num_rows == 4) {
                const SubFactor00 = self.data[2].v[2] * self.data[3].v[3] - self.data[3].v[2] * self.data[2].v[3];
                const SubFactor01 = self.data[2].v[1] * self.data[3].v[3] - self.data[3].v[1] * self.data[2].v[3];
                const SubFactor02 = self.data[2].v[1] * self.data[3].v[2] - self.data[3].v[1] * self.data[2].v[2];
                const SubFactor03 = self.data[2].v[0] * self.data[3].v[3] - self.data[3].v[0] * self.data[2].v[3];
                const SubFactor04 = self.data[2].v[0] * self.data[3].v[2] - self.data[3].v[0] * self.data[2].v[2];
                const SubFactor05 = self.data[2].v[0] * self.data[3].v[1] - self.data[3].v[0] * self.data[2].v[1];
                const DetCof = Vec(4, scalar_type).init(.{
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

        /// Inverse (GLM `inverse`; square 2x2/3x3/4x4 only): the matrix `M` with
        /// `M.mul(self) == identity()`. Computed via the cofactor
        /// expansion, divided by the determinant — an ill-conditioned
        /// (near-singular) matrix inverts to huge values. For rigid
        /// transforms prefer `affineInverse`, which skips most of the work.
        pub fn inverse(self: Self) Self {
            if (comptime num_columns != num_rows) @compileError("inverse requires a square matrix");
            if (comptime num_rows == 2) {
                const OneOverDeterminant = @as(scalar_type, 1) / (self.data[0].v[0] * self.data[1].v[1] - self.data[1].v[0] * self.data[0].v[1]);
                return init(.{
                    Vec(2, scalar_type).init(.{ self.data[1].v[1] * OneOverDeterminant, -self.data[0].v[1] * OneOverDeterminant }),
                    Vec(2, scalar_type).init(.{ -self.data[1].v[0] * OneOverDeterminant, self.data[0].v[0] * OneOverDeterminant }),
                });
            } else if (comptime num_rows == 3) {
                const t1 = self.data[1].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[1].v[2];
                const t2 = self.data[0].v[1] * self.data[2].v[2] - self.data[2].v[1] * self.data[0].v[2];
                const t3 = self.data[0].v[1] * self.data[1].v[2] - self.data[1].v[1] * self.data[0].v[2];
                const OneOverDeterminant = @as(scalar_type, 1) / (self.data[0].v[0] * t1 - self.data[1].v[0] * t2 + self.data[2].v[0] * t3);
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
            } else if (comptime num_rows == 4) {
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

                const Fac0 = Vec(4, scalar_type).init(.{ Coef00, Coef00, Coef02, Coef03 });
                const Fac1 = Vec(4, scalar_type).init(.{ Coef04, Coef04, Coef06, Coef07 });
                const Fac2 = Vec(4, scalar_type).init(.{ Coef08, Coef08, Coef10, Coef11 });
                const Fac3 = Vec(4, scalar_type).init(.{ Coef12, Coef12, Coef14, Coef15 });
                const Fac4 = Vec(4, scalar_type).init(.{ Coef16, Coef16, Coef18, Coef19 });
                const Fac5 = Vec(4, scalar_type).init(.{ Coef20, Coef20, Coef22, Coef23 });

                const Vec0 = Vec(4, scalar_type).init(.{ self.data[1].v[0], self.data[0].v[0], self.data[0].v[0], self.data[0].v[0] });
                const Vec1 = Vec(4, scalar_type).init(.{ self.data[1].v[1], self.data[0].v[1], self.data[0].v[1], self.data[0].v[1] });
                const Vec2 = Vec(4, scalar_type).init(.{ self.data[1].v[2], self.data[0].v[2], self.data[0].v[2], self.data[0].v[2] });
                const Vec3 = Vec(4, scalar_type).init(.{ self.data[1].v[3], self.data[0].v[3], self.data[0].v[3], self.data[0].v[3] });

                const Inv0 = Vec1.mul(Fac0).sub(Vec2.mul(Fac1)).add(Vec3.mul(Fac2));
                const Inv1 = Vec0.mul(Fac0).sub(Vec2.mul(Fac3)).add(Vec3.mul(Fac4));
                const Inv2 = Vec0.mul(Fac1).sub(Vec1.mul(Fac3)).add(Vec3.mul(Fac5));
                const Inv3 = Vec0.mul(Fac2).sub(Vec1.mul(Fac4)).add(Vec2.mul(Fac5));

                const SignA = Vec(4, scalar_type).init(.{ @as(scalar_type, 1), @as(scalar_type, -1), @as(scalar_type, 1), @as(scalar_type, -1) });
                const SignB = Vec(4, scalar_type).init(.{ @as(scalar_type, -1), @as(scalar_type, 1), @as(scalar_type, -1), @as(scalar_type, 1) });
                const Inverse = init(.{
                    Inv0.mul(SignA),
                    Inv1.mul(SignB),
                    Inv2.mul(SignA),
                    Inv3.mul(SignB),
                });

                const Row0 = Vec(4, scalar_type).init(.{ Inverse.data[0].v[0], Inverse.data[1].v[0], Inverse.data[2].v[0], Inverse.data[3].v[0] });
                const Dot0 = self.data[0].mul(Row0);
                const Dot1 = (Dot0.v[0] + Dot0.v[1]) + (Dot0.v[2] + Dot0.v[3]);

                const OneOverDeterminant = @as(scalar_type, 1) / Dot1;
                return Inverse.mulScalar(OneOverDeterminant);
            } else {
                @compileError("inverse only supported for 2x2, 3x3 and 4x4 matrices");
            }
        }

        // ---- transforms (4x4 only) ----

        /// Append a translation by `v` (GLM `translate(m, v)`): rebuilds
        /// column 3 as a linear combination of the existing columns, which
        /// makes the translation live in the LOCAL frame of `self`. Compose
        /// left-to-right: `matrix.identity().translate(t).rotate(θ, axis)` puts
        /// the rotation around the translated origin.
        pub fn translate(self: Self, translation: Vec(3, scalar_type)) Self {
            if (comptime num_columns != 4 or num_rows != 4) @compileError("translate requires a 4x4 matrix");
            var res = self;
            res.data[3] = self.data[0].mul(translation.v[0])
                .add(self.data[1].mul(translation.v[1]))
                .add(self.data[2].mul(translation.v[2]))
                .add(self.data[3]);
            return res;
        }

        /// Append a non-uniform scale by `v` (GLM `scale(m, v)`): multiplies the
        /// first three columns by the respective `v` components, keeping the
        /// translation column untouched, so the scale is applied in the
        /// local frame too. `scale(v)` with `v = (1,1,1)` is a no-op.
        pub fn scale(self: Self, scale_factors: Vec(3, scalar_type)) Self {
            if (comptime num_columns != 4 or num_rows != 4) @compileError("scale requires a 4x4 matrix");
            var res: Self = undefined;
            res.data[0] = self.data[0].mul(scale_factors.v[0]);
            res.data[1] = self.data[1].mul(scale_factors.v[1]);
            res.data[2] = self.data[2].mul(scale_factors.v[2]);
            res.data[3] = self.data[3];
            return res;
        }

        /// Append a rotation of `angle` radians around `axis` (GLM
        /// `rotate(m, angle, axis)`). The axis is normalized internally, its
        /// sign (and thus the rotation handedness) follows the input; the
        /// rotation is applied before `self`'s translation, i.e. around the
        /// origin of `self`. Counter-clockwise for positive angles around
        /// the positive axis (right-handed convention).
        pub fn rotate(self: Self, angle: scalar_type, axis: Vec(3, scalar_type)) Self {
            if (comptime num_columns != 4 or num_rows != 4) @compileError("rotate requires a 4x4 matrix");
            const c = scalar.cos(angle);
            const s = scalar.sin(angle);
            const axis_normal = axis.normalize();
            const temp = axis_normal.mul(@as(scalar_type, 1) - c);
            var rot = Self.zero();
            rot.data[0].v[0] = c + temp.v[0] * axis_normal.v[0];
            rot.data[0].v[1] = temp.v[0] * axis_normal.v[1] + s * axis_normal.v[2];
            rot.data[0].v[2] = temp.v[0] * axis_normal.v[2] - s * axis_normal.v[1];
            rot.data[1].v[0] = temp.v[1] * axis_normal.v[0] - s * axis_normal.v[2];
            rot.data[1].v[1] = c + temp.v[1] * axis_normal.v[1];
            rot.data[1].v[2] = temp.v[1] * axis_normal.v[2] + s * axis_normal.v[0];
            rot.data[2].v[0] = temp.v[2] * axis_normal.v[0] + s * axis_normal.v[1];
            rot.data[2].v[1] = temp.v[2] * axis_normal.v[1] - s * axis_normal.v[0];
            rot.data[2].v[2] = c + temp.v[2] * axis_normal.v[2];
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

        /// Convert to a 4x4 matrix (GLM `mat4(mat)` constructor behavior):
        /// smaller shapes are padded with zeroes and a `1` in the (3,3)
        /// slot, reproducing GLM's historical padding quirks exactly
        /// (verified against GLM 1.1 ref output). Use to lift a 2D/3D
        /// transform into homogeneous space.
        pub fn toMat4(self: Self) Mat(4, 4, scalar_type) {
            if (comptime num_columns == 4 and num_rows == 4) return self;
            const c0 = self.data[0];
            const c1 = self.data[1];
            if (comptime num_columns >= 3 and num_rows >= 3) {
                // 3x3 / 3x4 / 4x3
                if (comptime num_rows == 4) {
                    return Mat(4, 4, scalar_type).init(.{ c0, c1, self.data[2], Vec(4, scalar_type).init(.{ 0, 0, 0, 1 }) });
                } else if (comptime num_rows == 3) {
                    return Mat(4, 4, scalar_type).init(.{
                        Vec(4, scalar_type).init(.{ c0.v[0], c0.v[1], c0.v[2], 0 }),
                        Vec(4, scalar_type).init(.{ c1.v[0], c1.v[1], c1.v[2], 0 }),
                        Vec(4, scalar_type).init(.{ self.data[2].v[0], self.data[2].v[1], self.data[2].v[2], 0 }),
                        Vec(4, scalar_type).init(.{ 0, 0, 0, 1 }),
                    });
                } else {
                    @compileError("toMat4: unsupported shape");
                }
            } else if (comptime num_columns == 3 and num_rows == 2) {
                return Mat(4, 4, scalar_type).init(.{
                    Vec(4, scalar_type).init(.{ c0.v[0], c0.v[1], 0, 0 }),
                    Vec(4, scalar_type).init(.{ c1.v[0], c1.v[1], 0, 0 }),
                    Vec(4, scalar_type).init(.{ self.data[2].v[0], self.data[2].v[1], 1, 0 }),
                    Vec(4, scalar_type).init(.{ 0, 0, 0, 1 }),
                });
            } else if (comptime num_columns == 4 and num_rows == 2) {
                return Mat(4, 4, scalar_type).init(.{
                    Vec(4, scalar_type).init(.{ c0.v[0], c0.v[1], 0, 0 }),
                    Vec(4, scalar_type).init(.{ c1.v[0], c1.v[1], 0, 0 }),
                    Vec(4, scalar_type).init(.{ 0, 0, 1, 0 }),
                    Vec(4, scalar_type).init(.{ 0, 0, 0, 1 }),
                });
            } else if (comptime num_columns == 2) {
                const pad2 = Vec(4, scalar_type).init(.{ 0, 0, 1, 0 });
                const pad3 = Vec(4, scalar_type).init(.{ 0, 0, 0, 1 });
                if (comptime num_rows == 4) {
                    return Mat(4, 4, scalar_type).init(.{ c0, c1, pad2, pad3 });
                } else if (comptime num_rows == 3) {
                    return Mat(4, 4, scalar_type).init(.{
                        Vec(4, scalar_type).init(.{ c0.v[0], c0.v[1], c0.v[2], 0 }),
                        Vec(4, scalar_type).init(.{ c1.v[0], c1.v[1], c1.v[2], 0 }),
                        pad2,
                        pad3,
                    });
                } else {
                    return Mat(4, 4, scalar_type).init(.{
                        Vec(4, scalar_type).init(.{ c0.v[0], c0.v[1], 0, 0 }),
                        Vec(4, scalar_type).init(.{ c1.v[0], c1.v[1], 0, 0 }),
                        pad2,
                        pad3,
                    });
                }
            } else {
                @compileError("toMat4: unsupported shape");
            }
        }

        // ---- printing ----

        /// `{any}` formatter: prints elements in column-major order as
        /// `{m00,m10,m20,m30,m01,...}` with `{d}` float formatting.
        /// Enables `std.debug.print("{any}", .{m})` in logs and tests.
        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.writeByte('{');
            for (0..num_columns) |c| {
                for (0..num_rows) |r| {
                    try writer.print("{d},", .{self.data[c].v[r]});
                }
            }
            try writer.writeByte('}');
        }

        // ---- ext/matrix_common ----

        /// Per-component absolute value (GLM `abs(mat)`): no linear algebra
        /// meaning — just |element|, e.g. to compare transform magnitudes.
        pub inline fn abs(self: Self) Self {
            var res: Self = undefined;
            inline for (0..num_columns) |c| res.data[c] = self.data[c].abs();
            return res;
        }

        /// Blend between `self` and `right_hand_side` (GLM `mix(x, y, a)`):
        /// - `a` scalar: per-element `x·(1−a) + y·a` (the linear lerp),
        /// - `a` matrix: GLM's quirky elementwise formula
        ///   `x · (ones() − a) + y · a` — note the `1` is a full ones
        ///   matrix, NOT the identity, so off-diagonal elements blend too.
        /// Use the scalar form to fade between two keyframe transforms.
        pub fn mix(self: Self, right_hand_side: Self, factor: anytype) Self {
            const AT = @TypeOf(factor);
            if (comptime isMat(AT)) {
                const ones_minus = ones().sub(factor);
                return self.matrixCompMult(ones_minus).add(right_hand_side.matrixCompMult(factor));
            }
            const factor_value: scalar_type = scalar.cast(scalar_type, factor);
            var res: Self = undefined;
            inline for (0..num_columns) |c| res.data[c] = self.data[c].mul(@as(scalar_type, 1) - factor_value).add(right_hand_side.data[c].mul(factor_value));
            return res;
        }

        // ---- ext/matrix_relational ----

        /// Backend of the four tolerance comparisons: for each column c,
        /// bits[c] is true iff the column vectors `self.data[c]` and
        /// `right_hand_side.data[c]` agree (need_all) or differ (any) within the
        /// per-column tolerance `arg` — which may be a scalar or a vector
        /// whose component c holds column c's tolerance. `ulp` selects the
        /// ULP metric instead of the absolute epsilon.
        fn colCmp(self: Self, right_hand_side: Self, comptime need_all: bool, tolerance: anytype, comptime ulp: bool) Vec(num_columns, bool) {
            const AT = @TypeOf(tolerance);
            const av = comptime vec.isVec(AT);
            var res: @Vector(num_columns, bool) = undefined;
            inline for (0..num_columns) |c| {
                const a = self.data[c];
                const b = right_hand_side.data[c];
                const bit = if (comptime need_all) blk: {
                    if (comptime ulp) {
                        const e = a.equalULP(b, if (av) tolerance.v[c] else tolerance);
                        break :blk e.all();
                    } else {
                        const e = a.equalEps(b, if (av) tolerance.v[c] else tolerance);
                        break :blk e.all();
                    }
                } else blk: {
                    if (comptime ulp) {
                        const e = a.notEqualULP(b, if (av) tolerance.v[c] else tolerance);
                        break :blk e.any();
                    } else {
                        const e = a.notEqualEps(b, if (av) tolerance.v[c] else tolerance);
                        break :blk e.any();
                    }
                };
                res[c] = bit;
            }
            return .{ .v = res };
        }

        /// Exact per-column equality (GLM `equal(mat, mat)`): result column c is
        /// true iff ALL components of columns c are exactly equal (bitwise
        /// on floats). Intended for integer matrices; for float work use
        /// `equalEps`/`equalULP` — exact equality is almost never what you
        /// want after arithmetic.
        pub fn equal(self: Self, right_hand_side: Self) Vec(num_columns, bool) {
            var res: @Vector(num_columns, bool) = undefined;
            inline for (0..num_columns) |c| res[c] = self.data[c].equal(right_hand_side.data[c]).all();
            return .{ .v = res };
        }

        /// Per-column inequality (GLM `notEqual(mat, mat)`): result column c is
        /// true iff ANY component of column c differs. The negation of
        /// `equal`.
        pub fn notEqual(self: Self, right_hand_side: Self) Vec(num_columns, bool) {
            var res: @Vector(num_columns, bool) = undefined;
            inline for (0..num_columns) |c| res[c] = self.data[c].notEqual(right_hand_side.data[c]).any();
            return .{ .v = res };
        }

        /// Per-column tolerance equality (GLM `equal(mat, mat, eps)`): column c
        /// is true iff every element of column c differs by < `eps[c]`
        /// (eps vector) or < `eps` (scalar). The standard way to assert two
        /// transform matrices are "the same" after floating-point work.
        pub fn equalEps(self: Self, right_hand_side: Self, epsilon: anytype) Vec(num_columns, bool) {
            return self.colCmp(right_hand_side, true, epsilon, false);
        }

        /// Per-column tolerance inequality (GLM `notEqual(mat, mat, eps)`):
        /// column c is true iff some element of column c differs by at
        /// least the given tolerance.
        pub fn notEqualEps(self: Self, right_hand_side: Self, epsilon: anytype) Vec(num_columns, bool) {
            return self.colCmp(right_hand_side, false, epsilon, false);
        }

        /// Per-column ULP equality (GLM `equal(mat, mat, maxULPs)`): like
        /// `equalEps`, but the tolerance is measured in representable float
        /// steps, so it stays meaningful across exponent scales — use for
        /// numerically-generated matrices (e.g. iterative inverses).
        pub fn equalULP(self: Self, right_hand_side: Self, max_ulps: anytype) Vec(num_columns, bool) {
            return self.colCmp(right_hand_side, true, max_ulps, true);
        }

        /// Per-column ULP inequality (GLM `notEqual(mat, mat, maxULPs)`): the
        /// negation of `equalULP` — column c is true iff some element
        /// exceeds the ULP budget.
        pub fn notEqualULP(self: Self, right_hand_side: Self, max_ulps: anytype) Vec(num_columns, bool) {
            return self.colCmp(right_hand_side, false, max_ulps, true);
        }
    };
}

fn comptimePrint(comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.comptimePrint(fmt, args);
}

// ---- free functions (GLM ext/matrix_transform, ext/matrix_clip_space) ----

/// The common matrix type used by all free functions below, matching
/// GLM's default `glm::mat4` (4 columns, 4 rows, f32).
const mat4 = Mat(4, 4, f32);

/// Build a right-handed view matrix looking from `eye` toward `center`
/// (GLM `lookAt`, default clip control RH with NDC in [-1, 1]): the
/// basis is f = normalize(center−eye), s = f×up (side), u = s×f (up),
/// then the eye is expressed in that basis. Use for the camera matrix of
/// OpenGL-style renderers.
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

/// Left-handed twin of `lookAt` (GLM `lookAtLH`, NDC in [-1, 1]): the
/// side vector is `up×f` and the forward axis keeps its sign, so
/// the z-axis points INTO the scene — use for DirectX-style (or
/// flipped-z) camera conventions.
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

/// Perspective projection (GLM `perspective`, default = right-handed,
/// NDC in [-1, 1]): maps a symmetric frustum with vertical field of
/// view `fovy` (radians) and `aspect = width/height` into clip
/// space, with the near plane at z = −zNear. This is the usual
/// OpenGL camera projection; Vulkan/Metal want the ZO variant.
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

/// Right-handed perspective with NDC z in [0, 1] (GLM
/// `perspectiveRH_ZO`): the depth output of Vulkan and Metal
/// (D3D-style) pipelines, so the depth buffer contains positive z
/// in [0, 1] with a reverse-friendly layout.
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

/// Left-handed perspective with NDC z in [0, 1] (GLM
/// `perspectiveLH_ZO`): z points into the scene, positive depth —
/// the combination used by many engines that flip z to get
/// right-handed rendering with a D3D-style depth range.
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

/// Orthographic projection (GLM `ortho`, default = right-handed, NDC
/// [-1, 1]): maps the axis-aligned box [left,right]×[bottom,top]×
/// [zNear,zFar] linearly into clip space without perspective
/// division — use for 3D UI, minimaps, and 2D overlays (with
/// `ortho2D` when no depth range matters).
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

/// Right-handed ortho with NDC z in [0, 1] (GLM `orthoRH_ZO`): the
/// depth half-range is compressed to [0, 1] with the near plane at
/// z = 0 — pair with `perspectiveRH_ZO` for consistent depth
/// conventions in a D3D-style renderer.
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

/// Frustum projection (GLM `frustum`, default = right-handed, NDC
/// [-1, 1]): the general asymmetric perspective, defined by an
/// arbitrary near clipping rectangle instead of a symmetric fov.
/// Concave/degenerate boxes (left ≥ right etc.) produce singular
/// matrices; `perspective` is the common specialization.
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

/// Explicit right-handed view (GLM `lookAtRH`): GLM's default clip
/// control is RH + NDC [-1, 1], so this is numerically identical
/// to `lookAt` — kept for parity with GLM's API surface.
pub fn lookAtRH(eye: Vec(3, f32), center: Vec(3, f32), up: Vec(3, f32)) mat4 {
    return lookAt(eye, center, up);
}

/// Right-handed perspective with NDC z in [-1, 1] (GLM
/// `perspectiveRH_NO`): GLM's default, therefore identical to
/// `perspective`; use OpenGL-style depth as usual.
pub fn perspectiveRH_NO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspective(fovy, aspect, zNear, zFar);
}

/// Left-handed perspective with NDC z in [-1, 1] (GLM
/// `perspectiveLH_NO`): forward is +z (into the scene) while depth
/// keeps OpenGL's symmetric range — matches the flipped-z trick
/// engines use for better precision without going ZO.
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

/// Zoom-variant perspective (GLM `perspectiveZO`; default clip control
/// is RH so this equals `perspectiveRH_ZO`). Named for the NDC
/// [0, 1] depth range; use in Vulkan/Metal pipelines.
pub fn perspectiveZO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveRH_ZO(fovy, aspect, zNear, zFar);
}

/// Negative-to-one NDC perspective (GLM `perspectiveNO`; equals
/// `perspective`, the OpenGL default). The "NO" = [-1, 1] depth
/// range convention that GLM's aliases default to.
pub fn perspectiveNO(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspective(fovy, aspect, zNear, zFar);
}

/// Left-handed perspective alias (GLM `perspectiveLH`; GLM's default
/// LH clip control is NO, so this equals `perspectiveLH_NO`).
pub fn perspectiveLH(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveLH_NO(fovy, aspect, zNear, zFar);
}

/// Right-handed perspective alias (GLM `perspectiveRH`; GLM's default
/// RH clip control is NO, so this equals `perspective`).
pub fn perspectiveRH(fovy: f32, aspect: f32, zNear: f32, zFar: f32) mat4 {
    return perspective(fovy, aspect, zNear, zFar);
}

/// Right-handed fov-from-size perspective with NDC z in [0, 1] (GLM
/// `perspectiveFovRH_ZO`): `fov` is the total vertical field of
/// view in radians, and the frustum is derived from a pixel
/// `width` × `height` instead of an aspect ratio — use for
/// render-to-texture cameras that must match a specific viewport.
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

/// Right-handed fov-from-size perspective with NDC z in [-1, 1] (GLM
/// `perspectiveFovRH_NO`): like `perspectiveFovRH_ZO` but with
/// OpenGL's symmetric depth range.
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

/// Left-handed fov-from-size perspective with NDC z in [0, 1] (GLM
/// `perspectiveFovLH_ZO`): LH counterpart of `perspectiveFovRH_ZO`
/// with positive depth — matches D3D12/Metal depth ranges.
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

/// Left-handed fov-from-size perspective with NDC z in [-1, 1] (GLM
/// `perspectiveFovLH_NO`): LH twin of `perspectiveFovRH_NO`.
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

/// Fov-from-size perspective default (GLM `perspectiveFov`; RH + NDC
/// [-1, 1], so identical to `perspectiveFovRH_NO`).
pub fn perspectiveFov(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
}

/// Fov-from-size perspective, zoom depth range (GLM `perspectiveFovZO`;
/// equals `perspectiveFovRH_ZO` under GLM's default RH control).
pub fn perspectiveFovZO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_ZO(fov, width, height, zNear, zFar);
}

/// Fov-from-size perspective, negative-to-one depth (GLM
/// `perspectiveFovNO`; equals `perspectiveFovRH_NO`).
pub fn perspectiveFovNO(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
}

/// Fov-from-size perspective, left-handed (GLM `perspectiveFovLH`;
/// GLM's LH default is NO, so this equals `perspectiveFovLH_NO`).
pub fn perspectiveFovLH(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovLH_NO(fov, width, height, zNear, zFar);
}

/// Fov-from-size perspective, right-handed (GLM `perspectiveFovRH`;
/// equals `perspectiveFovRH_NO` under GLM's RH default).
pub fn perspectiveFovRH(fov: f32, width: f32, height: f32, zNear: f32, zFar: f32) mat4 {
    return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
}

/// Infinite-far-plane perspective, right-handed, NDC [-1, 1] (GLM
/// `infinitePerspectiveRH_NO`): zFar is dropped, the depth
/// function maps zNear to the far end of the range and everything
/// beyond it — good for starfields, large outdoor scenes and
/// reverse-z setups that must never clip at a finite distance.
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

/// Infinite-far-plane perspective, right-handed, NDC [0, 1] (GLM
/// `infinitePerspectiveRH_ZO`): like the NO variant but with
/// Vulkan/Metal depth conventions; note the depth at zNear is 0,
/// so a reverse-z depth buffer pairs naturally.
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

/// Infinite-far-plane perspective, left-handed, NDC [-1, 1] (GLM
/// `infinitePerspectiveLH_NO`): LH twin of the RH_NO variant, for
/// +z-forward engines that need unlimited draw distance.
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

/// Infinite-far-plane perspective, left-handed, NDC [0, 1] (GLM
/// `infinitePerspectiveLH_ZO`): LH + ZO combination, e.g. for
/// D3D12-style pipelines with unlimited far distance.
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

/// Infinite perspective default (GLM `infinitePerspective`; RH + NDC
/// [-1, 1], identical to `infinitePerspectiveRH_NO`).
pub fn infinitePerspective(fovy: f32, aspect: f32, zNear: f32) mat4 {
    return infinitePerspectiveRH_NO(fovy, aspect, zNear);
}

/// Infinite perspective with Lengyel's tweak (GLM
/// `tweakedInfinitePerspective(fovy, aspect, zNear, ep)`): the
/// `ep` parameter (usually machine epsilon) replaces the far
/// plane, removing the depth compression singularity that plain
/// infinite projections suffer at z → zNear.
pub fn tweakedInfinitePerspective(fovy: f32, aspect: f32, zNear: f32, epsilon: f32) mat4 {
    const range = scalar.tan(fovy / 2) * zNear;
    const left = -range * aspect;
    const right = range * aspect;
    const bottom = -range;
    const top = range;
    var res = mat4.zero();
    res.data[0].v[0] = (2 * zNear) / (right - left);
    res.data[1].v[1] = (2 * zNear) / (top - bottom);
    res.data[2].v[2] = epsilon - 1;
    res.data[2].v[3] = -1;
    res.data[3].v[2] = (epsilon - 2) * zNear;
    return res;
}

/// `tweakedInfinitePerspective` with `ep = f32 epsilon` (GLM
/// `tweakedInfinitePerspectiveDefault`): the recommended instant;
/// just pass fov/aspect/zNear.
pub fn tweakedInfinitePerspectiveDefault(fovy: f32, aspect: f32, zNear: f32) mat4 {
    return tweakedInfinitePerspective(fovy, aspect, zNear, std.math.floatEps(f32));
}

/// 2D orthographic projection (GLM `ortho(left, right, bottom, top)`):
/// the z axis is left alone (depth -1..1 as identity) — the
/// standard matrix for flat 2D rendering, UI overlays and screen
/// space coordinates in pixels.
pub fn ortho2D(left: f32, right: f32, bottom: f32, top: f32) mat4 {
    var res = mat4.identity();
    res.data[0].v[0] = 2 / (right - left);
    res.data[1].v[1] = 2 / (top - bottom);
    res.data[2].v[2] = -1;
    res.data[3].v[0] = -(right + left) / (right - left);
    res.data[3].v[1] = -(top + bottom) / (top - bottom);
    return res;
}

/// Left-handed ortho with NDC z in [0, 1] (GLM `orthoLH_ZO`): depth
/// maps zNear → 0, zFar → 1 with +z forward — the D3D12-style
/// ortho to pair with `perspectiveLH_ZO`.
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

/// Left-handed ortho with NDC z in [-1, 1] (GLM `orthoLH_NO`): +z
/// forward with OpenGL's symmetric depth range.
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

/// Right-handed ortho with NDC z in [-1, 1] (GLM `orthoRH_NO`): the
/// OpenGL-style default, identical math to `ortho` itself.
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

/// Zoom-depth ortho (GLM `orthoZO`; equals `orthoRH_ZO` under GLM's RH
/// default). Alias for parity with the ZO/NO naming.
pub fn orthoZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoRH_ZO(left, right, bottom, top, zNear, zFar);
}

/// Negative-to-one-depth ortho (GLM `orthoNO`; equals `orthoRH_NO`,
/// the OpenGL default). Alias for parity with the ZO/NO naming.
pub fn orthoNO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoRH_NO(left, right, bottom, top, zNear, zFar);
}

/// Left-handed ortho alias (GLM `orthoLH`; GLM's LH default is NO, so
/// this equals `orthoLH_NO`).
pub fn orthoLH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoLH_NO(left, right, bottom, top, zNear, zFar);
}

/// Right-handed ortho alias (GLM `orthoRH`; equals `orthoRH_NO`, the
/// OpenGL default).
pub fn orthoRH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return orthoRH_NO(left, right, bottom, top, zNear, zFar);
}

/// Left-handed asymmetric frustum with NDC z in [0, 1] (GLM
/// `frustumLH_ZO`): +z forward, zNear → 0. Useful for asymmetric
/// near planes (e.g. off-axis projection) in D3D-style pipelines.
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

/// Left-handed asymmetric frustum with NDC z in [-1, 1] (GLM
/// `frustumLH_NO`): +z forward with OpenGL's symmetric depth.
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

/// Right-handed asymmetric frustum with NDC z in [0, 1] (GLM
/// `frustumRH_ZO`): −z forward, zNear → 0 — Vulkan/Metal variant
/// for off-axis projections (CAVE/portal rendering).
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

/// Right-handed asymmetric frustum with NDC z in [-1, 1] (GLM
/// `frustumRH_NO`): the OpenGL default, identical math to
/// `frustum`.
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

/// Zoom-depth frustum (GLM `frustumZO`; equals `frustumRH_ZO` under RH
/// default). Alias for ZO/NO parity.
pub fn frustumZO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumRH_ZO(left, right, bottom, top, zNear, zFar);
}

/// Negative-to-one-depth frustum (GLM `frustumNO`; equals
/// `frustumRH_NO`). Alias for ZO/NO parity.
pub fn frustumNO(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumRH_NO(left, right, bottom, top, zNear, zFar);
}

/// Left-handed frustum alias (GLM `frustumLH`; GLM's LH default is NO,
/// so this equals `frustumLH_NO`).
pub fn frustumLH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumLH_NO(left, right, bottom, top, zNear, zFar);
}

/// Right-handed frustum alias (GLM `frustumRH`; equals
/// `frustumRH_NO`, the OpenGL default).
pub fn frustumRH(left: f32, right: f32, bottom: f32, top: f32, zNear: f32, zFar: f32) mat4 {
    return frustumRH_NO(left, right, bottom, top, zNear, zFar);
}

// ---- GLM ext/matrix_projection ----

/// Project a world-space point to window space with NDC z in
/// [0, 1] (GLM `projectZO(obj, model, proj, viewport)`):
/// `obj` is transformed by `model * proj`, mapped through the
/// viewport rect (x, y, width, height) and returned as screen
/// coordinates with depth in [0, 1]. Note GLM/MathML: with ZO
/// projections the returned z is clip-relative, not viewport-raw.
pub fn projectZO(obj: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    var tmp = proj.mulVec(model.mulVec(Vec(4, f32).init(.{ obj.v[0], obj.v[1], obj.v[2], 1 })));
    tmp = tmp.div(tmp.v[3]);
    tmp.v[0] = tmp.v[0] * 0.5 + 0.5;
    tmp.v[1] = tmp.v[1] * 0.5 + 0.5;
    tmp.v[0] = tmp.v[0] * viewport.v[2] + viewport.v[0];
    tmp.v[1] = tmp.v[1] * viewport.v[3] + viewport.v[1];
    return Vec(3, f32).init(.{ tmp.v[0], tmp.v[1], tmp.v[2] });
}

/// Project a world-space point to window space with NDC z in [-1, 1]
/// (GLM `projectNO(obj, model, proj, viewport)`): the OpenGL
/// counterpart of `projectZO` — screen x/y (pixels) plus the raw
/// viewport-unmapped depth in NDC [-1, 1]. Feed the result of
/// `project`/`projectNO` into `unProjectNO` to round-trip.
pub fn projectNO(obj: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    var tmp = proj.mulVec(model.mulVec(Vec(4, f32).init(.{ obj.v[0], obj.v[1], obj.v[2], 1 })));
    tmp = tmp.div(tmp.v[3]);
    tmp = tmp.mul(0.5).add(0.5);
    tmp.v[0] = tmp.v[0] * viewport.v[2] + viewport.v[0];
    tmp.v[1] = tmp.v[1] * viewport.v[3] + viewport.v[1];
    return Vec(3, f32).init(.{ tmp.v[0], tmp.v[1], tmp.v[2] });
}

/// Projection default (GLM `project`; GLM's default clip control is NO,
/// so this equals `projectNO`) — world point → screen pixels +
/// OpenGL-style depth.
pub fn project(obj: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    return projectNO(obj, model, proj, viewport);
}

/// Unproject a window-space point to world space with NDC z in [0, 1]
/// (GLM `unProjectZO(win, model, proj, viewport)`): the inverse of
/// `projectZO` — invert the `proj * model` chain, undo the viewport
/// mapping and perspective division. Use for mouse picking in
/// Vulkan/Metal-style pipelines.
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

/// Unproject a window-space point to world space with NDC z in [-1, 1]
/// (GLM `unProjectNO(win, model, proj, viewport)`): the OpenGL
/// counterpart of `unProjectZO` — use for mouse picking with
/// OpenGL-style depth buffers.
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

/// Unprojection default (GLM `unProject`; equals `unProjectNO` under
/// GLM's NO default).
pub fn unProject(win: Vec(3, f32), model: mat4, proj: mat4, viewport: Vec(4, f32)) Vec(3, f32) {
    return unProjectNO(win, model, proj, viewport);
}

/// Build a picking matrix (GLM `pickMatrix(center, delta, viewport)`):
/// a translate+scale that isolates the `delta`×`delta` region
/// around `center` (both in window pixels) as the new frustum —
/// combine with rendering a second pass of the scene to implement
/// rubber-band selection. Returns identity if delta is not
/// positive.
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

/// Inverse of a 3x3 affine (rotation+translation) matrix (GLM
/// `affineInverse`): the upper-left 2x2 is inverted in isolation
/// and the translation column of the result is placed in the third
/// column; the bottom row is implicitly [0, 0, 1]. Only valid when
/// the matrix really is affine — cheaper than a full `inverse`.
pub fn affineInverse3(matrix: Mat(3, 3, f32)) Mat(3, 3, f32) {
    const inv22 = Mat(2, 2, f32).init(.{
        Vec(2, f32).init(.{ matrix.data[0].v[0], matrix.data[0].v[1] }),
        Vec(2, f32).init(.{ matrix.data[1].v[0], matrix.data[1].v[1] }),
    }).inverse();
    const t = Vec(2, f32).init(.{ matrix.data[2].v[0], matrix.data[2].v[1] });
    const neg = inv22.mulVec(t).neg();
    return Mat(3, 3, f32).init(.{
        Vec(3, f32).init(.{ inv22.data[0].v[0], inv22.data[0].v[1], 0 }),
        Vec(3, f32).init(.{ inv22.data[1].v[0], inv22.data[1].v[1], 0 }),
        Vec(3, f32).init(.{ neg.v[0], neg.v[1], 1 }),
    });
}

/// Inverse of a 4x4 affine matrix (GLM `affineInverse`): inverts the
/// upper-left 3x3 and folds the translation in as `−num_rows⁻¹·t`; bottom
/// row stays [0, 0, 0, 1]. The right tool for view/model matrices
/// (rigid body transforms) — an order of magnitude less work than
/// the full cofactor `inverse`.
pub fn affineInverse(matrix: mat4) mat4 {
    const inv33 = Mat(3, 3, f32).init(.{
        Vec(3, f32).init(.{ matrix.data[0].v[0], matrix.data[0].v[1], matrix.data[0].v[2] }),
        Vec(3, f32).init(.{ matrix.data[1].v[0], matrix.data[1].v[1], matrix.data[1].v[2] }),
        Vec(3, f32).init(.{ matrix.data[2].v[0], matrix.data[2].v[1], matrix.data[2].v[2] }),
    }).inverse();
    const t = Vec(3, f32).init(.{ matrix.data[3].v[0], matrix.data[3].v[1], matrix.data[3].v[2] });
    const neg = inv33.mulVec(t).neg();
    return mat4.init(.{
        Vec(4, f32).init(.{ inv33.data[0].v[0], inv33.data[0].v[1], inv33.data[0].v[2], 0 }),
        Vec(4, f32).init(.{ inv33.data[1].v[0], inv33.data[1].v[1], inv33.data[1].v[2], 0 }),
        Vec(4, f32).init(.{ inv33.data[2].v[0], inv33.data[2].v[1], inv33.data[2].v[2], 0 }),
        Vec(4, f32).init(.{ neg.v[0], neg.v[1], neg.v[2], 1 }),
    });
}

/// 2x2 inverse-transpose (GLM `inverseTranspose`, gtc/matrix_inverse):
/// NOTE — in GLM 1.1.0 this function returns the plain inverse for
/// 2x2 (the transpose step is missing upstream), and this port
/// deliberately reproduces GLM's output. For the mathematically
/// correct normal matrix of a 2D transform, transpose the result.
pub fn inverseTranspose2(matrix: Mat(2, 2, f32)) Mat(2, 2, f32) {
    const d = matrix.determinant();
    return Mat(2, 2, f32).init(.{
        Vec(2, f32).init(.{ matrix.data[1].v[1] / d, -matrix.data[0].v[1] / d }),
        Vec(2, f32).init(.{ -matrix.data[1].v[0] / d, matrix.data[0].v[0] / d }),
    });
}

/// 3x3 inverse-transpose (GLM `inverseTranspose`, gtc/matrix_inverse):
/// `(M⁻¹)ᵀ`, computed directly as the transposed cofactor matrix
/// divided by the determinant. This is the "normal matrix" to
/// transform surface normals so they stay perpendicular after a
/// non-uniform scale.
pub fn inverseTranspose3(matrix: Mat(3, 3, f32)) Mat(3, 3, f32) {
    const determinant =
        matrix.data[0].v[0] * (matrix.data[1].v[1] * matrix.data[2].v[2] - matrix.data[1].v[2] * matrix.data[2].v[1]) -
        matrix.data[0].v[1] * (matrix.data[1].v[0] * matrix.data[2].v[2] - matrix.data[1].v[2] * matrix.data[2].v[0]) +
        matrix.data[0].v[2] * (matrix.data[1].v[0] * matrix.data[2].v[1] - matrix.data[1].v[1] * matrix.data[2].v[0]);
    var inv: Mat(3, 3, f32) = undefined;
    inv.data[0].v[0] = (matrix.data[1].v[1] * matrix.data[2].v[2] - matrix.data[2].v[1] * matrix.data[1].v[2]);
    inv.data[0].v[1] = -(matrix.data[1].v[0] * matrix.data[2].v[2] - matrix.data[2].v[0] * matrix.data[1].v[2]);
    inv.data[0].v[2] = (matrix.data[1].v[0] * matrix.data[2].v[1] - matrix.data[2].v[0] * matrix.data[1].v[1]);
    inv.data[1].v[0] = -(matrix.data[0].v[1] * matrix.data[2].v[2] - matrix.data[2].v[1] * matrix.data[0].v[2]);
    inv.data[1].v[1] = (matrix.data[0].v[0] * matrix.data[2].v[2] - matrix.data[2].v[0] * matrix.data[0].v[2]);
    inv.data[1].v[2] = -(matrix.data[0].v[0] * matrix.data[2].v[1] - matrix.data[2].v[0] * matrix.data[0].v[1]);
    inv.data[2].v[0] = (matrix.data[0].v[1] * matrix.data[1].v[2] - matrix.data[1].v[1] * matrix.data[0].v[2]);
    inv.data[2].v[1] = -(matrix.data[0].v[0] * matrix.data[1].v[2] - matrix.data[1].v[0] * matrix.data[0].v[2]);
    inv.data[2].v[2] = (matrix.data[0].v[0] * matrix.data[1].v[1] - matrix.data[1].v[0] * matrix.data[0].v[1]);
    var res = inv;
    inline for (0..3) |c| {
        inline for (0..3) |r| res.data[c].v[r] /= determinant;
    }
    return res;
}

/// 4x4 inverse-transpose (GLM `inverseTranspose`, gtc/matrix_inverse):
/// `(M⁻¹)ᵀ` via the transposed cofactor expansion divided by the
/// determinant — the standard normal matrix for 3D rendering with
/// skewed or non-uniformly scaled models.
pub fn inverseTranspose(matrix: mat4) mat4 {
    const sf00 = matrix.data[2].v[2] * matrix.data[3].v[3] - matrix.data[3].v[2] * matrix.data[2].v[3];
    const sf01 = matrix.data[2].v[1] * matrix.data[3].v[3] - matrix.data[3].v[1] * matrix.data[2].v[3];
    const sf02 = matrix.data[2].v[1] * matrix.data[3].v[2] - matrix.data[3].v[1] * matrix.data[2].v[2];
    const sf03 = matrix.data[2].v[0] * matrix.data[3].v[3] - matrix.data[3].v[0] * matrix.data[2].v[3];
    const sf04 = matrix.data[2].v[0] * matrix.data[3].v[2] - matrix.data[3].v[0] * matrix.data[2].v[2];
    const sf05 = matrix.data[2].v[0] * matrix.data[3].v[1] - matrix.data[3].v[0] * matrix.data[2].v[1];
    const sf06 = matrix.data[1].v[2] * matrix.data[3].v[3] - matrix.data[3].v[2] * matrix.data[1].v[3];
    const sf07 = matrix.data[1].v[1] * matrix.data[3].v[3] - matrix.data[3].v[1] * matrix.data[1].v[3];
    const sf08 = matrix.data[1].v[1] * matrix.data[3].v[2] - matrix.data[3].v[1] * matrix.data[1].v[2];
    const sf09 = matrix.data[1].v[0] * matrix.data[3].v[3] - matrix.data[3].v[0] * matrix.data[1].v[3];
    const sf10 = matrix.data[1].v[0] * matrix.data[3].v[2] - matrix.data[3].v[0] * matrix.data[1].v[2];
    const sf11 = matrix.data[1].v[0] * matrix.data[3].v[1] - matrix.data[3].v[0] * matrix.data[1].v[1];
    const sf12 = matrix.data[1].v[2] * matrix.data[2].v[3] - matrix.data[2].v[2] * matrix.data[1].v[3];
    const sf13 = matrix.data[1].v[1] * matrix.data[2].v[3] - matrix.data[2].v[1] * matrix.data[1].v[3];
    const sf14 = matrix.data[1].v[1] * matrix.data[2].v[2] - matrix.data[2].v[1] * matrix.data[1].v[2];
    const sf15 = matrix.data[1].v[0] * matrix.data[2].v[3] - matrix.data[2].v[0] * matrix.data[1].v[3];
    const sf16 = matrix.data[1].v[0] * matrix.data[2].v[2] - matrix.data[2].v[0] * matrix.data[1].v[2];
    const sf17 = matrix.data[1].v[0] * matrix.data[2].v[1] - matrix.data[2].v[0] * matrix.data[1].v[1];

    var inv: mat4 = undefined;
    inv.data[0].v[0] = matrix.data[1].v[1] * sf00 - matrix.data[1].v[2] * sf01 + matrix.data[1].v[3] * sf02;
    inv.data[0].v[1] = -(matrix.data[1].v[0] * sf00 - matrix.data[1].v[2] * sf03 + matrix.data[1].v[3] * sf04);
    inv.data[0].v[2] = matrix.data[1].v[0] * sf01 - matrix.data[1].v[1] * sf03 + matrix.data[1].v[3] * sf05;
    inv.data[0].v[3] = -(matrix.data[1].v[0] * sf02 - matrix.data[1].v[1] * sf04 + matrix.data[1].v[2] * sf05);

    inv.data[1].v[0] = -(matrix.data[0].v[1] * sf00 - matrix.data[0].v[2] * sf01 + matrix.data[0].v[3] * sf02);
    inv.data[1].v[1] = matrix.data[0].v[0] * sf00 - matrix.data[0].v[2] * sf03 + matrix.data[0].v[3] * sf04;
    inv.data[1].v[2] = -(matrix.data[0].v[0] * sf01 - matrix.data[0].v[1] * sf03 + matrix.data[0].v[3] * sf05);
    inv.data[1].v[3] = matrix.data[0].v[0] * sf02 - matrix.data[0].v[1] * sf04 + matrix.data[0].v[2] * sf05;

    inv.data[2].v[0] = matrix.data[0].v[1] * sf06 - matrix.data[0].v[2] * sf07 + matrix.data[0].v[3] * sf08;
    inv.data[2].v[1] = -(matrix.data[0].v[0] * sf06 - matrix.data[0].v[2] * sf09 + matrix.data[0].v[3] * sf10);
    inv.data[2].v[2] = matrix.data[0].v[0] * sf07 - matrix.data[0].v[1] * sf09 + matrix.data[0].v[3] * sf11;
    inv.data[2].v[3] = -(matrix.data[0].v[0] * sf08 - matrix.data[0].v[1] * sf10 + matrix.data[0].v[2] * sf11);

    inv.data[3].v[0] = -(matrix.data[0].v[1] * sf12 - matrix.data[0].v[2] * sf13 + matrix.data[0].v[3] * sf14);
    inv.data[3].v[1] = matrix.data[0].v[0] * sf12 - matrix.data[0].v[2] * sf15 + matrix.data[0].v[3] * sf16;
    inv.data[3].v[2] = -(matrix.data[0].v[0] * sf13 - matrix.data[0].v[1] * sf15 + matrix.data[0].v[3] * sf17);
    inv.data[3].v[3] = matrix.data[0].v[0] * sf14 - matrix.data[0].v[1] * sf16 + matrix.data[0].v[2] * sf17;

    const determinant =
        matrix.data[0].v[0] * inv.data[0].v[0] +
        matrix.data[0].v[1] * inv.data[0].v[1] +
        matrix.data[0].v[2] * inv.data[0].v[2] +
        matrix.data[0].v[3] * inv.data[0].v[3];

    var res = inv;
    inline for (0..4) |c| {
        inline for (0..4) |r| res.data[c].v[r] /= determinant;
    }
    return res;
}
// ---- GLM func_common `outerProduct` ----

/// Outer product (GLM/GLSL `outerProduct(c, r)`): column vector `c`
/// times row vector `r`, yielding the matrix `res[j][i] =
/// c[i]*r[j]` — shaped `Mat(r.len, c.len)`, i.e. `c.len` rows and
/// `r.len` columns. The dual of the dot product: builds a rank-1
/// matrix such as the projector `outerProduct(n, n)` or the dyad
/// `I − 2·outerProduct(n, n)` used in reflections.
pub fn outerProduct(col_vec: anytype, row_vec: anytype) Mat(@TypeOf(row_vec).len, @TypeOf(col_vec).len, @TypeOf(col_vec).value_type) {
    const scalar_type = @TypeOf(col_vec).value_type;
    const CL = @TypeOf(col_vec).len;
    const RL = @TypeOf(row_vec).len;
    var res: Mat(RL, CL, scalar_type) = undefined;
    inline for (0..RL) |i| {
        inline for (0..CL) |j| res.data[i].v[j] = col_vec.v[j] * row_vec.v[i];
    }
    return res;
}
