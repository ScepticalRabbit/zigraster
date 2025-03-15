const std = @import("std");
const print = std.debug.print;
const expectEqual = std.testing.expectEqual;

const vector = @import("vector.zig");
const Vector = vector.Vector;
const Vec2f = vector.Vec2f;
const Vec3f = vector.Vec3f;

const EType = f64;
pub const Mat22f = Mat22T(f64);
pub const Mat33f = Mat33T(f64);
pub const Mat44f = Mat44T(f64);

pub fn Matrix(comptime rows_n_in: comptime_int, comptime cols_n_in: comptime_int, comptime ElemType: type) type {
    return extern struct {
        elems: [elem_n]ElemType,
        // rows_num: usize = rows_n_in,
        // cols_num: usize = cols_n_in,
        // elem_num: usize = elem_n,

        pub const rows_n: usize = rows_n_in;
        pub const cols_n: usize = cols_n_in;
        pub const elem_n: usize = rows_n_in * cols_n_in;

        const Self: type = @This();

        pub fn initFill(fill_val: ElemType) Self {
            return .{ .elems = [_]ElemType{fill_val} ** elem_n };
        }

        pub fn initZeros() Self {
            return initFill(0);
        }

        pub fn initOnes() Self {
            return initFill(1);
        }

        pub fn initIdentity() Self {
            var ident: Self = initZeros();

            // TODO: fix this for non-square matrices
            for (0..rows_n) |ii| {
                ident.set(ii, ii, 1);
            }

            return ident;
        }

        pub fn initDiag(diag_val: ElemType) Self {
            var ident: Self = initZeros();

            // TODO: fix this for non-square matrices
            for (0..rows_n) |ii| {
                ident.set(ii, ii, diag_val);
            }

            return ident;
        }

        pub fn initSlice(slice: []const ElemType) Self {
            return .{ .elems = slice[0..elem_n].* };
        }

        pub fn get(self: *const Self, row: usize, col: usize) ElemType {
            return self.elems[(row * cols_n) + col];
        }

        pub fn set(self: *Self, row: usize, col: usize, val: ElemType) void {
            self.elems[(row * cols_n) + col] = val;
        }

        pub fn getRowVec(self: *const Self, row: usize) Vector(cols_n,ElemType) {
            // TODO: make this bounds check?
            const start: usize = row*cols_n;
            const end: usize = start + cols_n;
            const row_slice: []const ElemType = self.elems[start..end];
            const vec = Vector(cols_n,ElemType).initSlice(row_slice);
            return vec;
        }

        pub fn getColVec(self: *const Self, col: usize) Vector(rows_n,ElemType) {
            // TODO: make this bounds check?
            var col_vec: [rows_n]ElemType = undefined;
            for (0..rows_n) |rr| {
                col_vec[rr] = self.get(rr,col);
            }
            const vec = Vector(rows_n,ElemType).initSlice(&col_vec);
            return vec;
        }

        pub fn getSubMat(self: *const Self, comptime rows: usize, comptime cols: usize, row_start: usize, col_start: usize) Matrix(rows,cols,ElemType) {
            // TODO: make this bounds check?
            const sub_mat = Matrix(rows,cols,ElemType).initZeros();

            for (row_start..rows) |rr| {
                for (col_start..cols) |cc| {
                    sub_mat.set(rr,cc,self.get(rr,cc));
                }
            }

            return sub_mat;
        }

        pub fn transpose(self: *const Self) Self {
            var mat_out: Self = undefined;

            for (0..rows_n) |ii| {
                for (ii..cols_n) |jj| {
                    mat_out.set(ii, jj, self.get(jj, ii));
                    mat_out.set(jj, ii, self.get(ii, jj));
                }
            }

            return mat_out;
        }

        pub fn trace(self: *const Self) ElemType {
            var trace_out: ElemType = 0;

            // TODO: fix this for non-square matrices
            for (0..rows_n) |ii| {
                trace_out += self.get(ii,ii);
            }

            return trace_out;
        }

        pub fn add(self: *const Self, to_add: Self) Self {
            var mat_out: Self = undefined;

            for (0..elem_n) |ee| {
                mat_out.elems[ee] = self.elems[ee] + to_add.elems[ee];
            }

            return mat_out;
        }

        pub fn subtract(self: *const Self, to_sub: Self) Self {
            var mat_out: Self = undefined;

            for (0..elem_n) |ee| {
                mat_out.elems[ee] = self.elems[ee] - to_sub.elems[ee];
            }

            return mat_out;
        }

        pub fn multScalar(self: *const Self, scalar: ElemType) Self {
            var mat_out: Self = undefined;

            for (0..elem_n) |ee| {
                mat_out.elems[ee] = scalar * self.elems[ee];
            }

            return mat_out;
        }

        pub fn multVec(self: *const Self, vec: Vector(cols_n, ElemType)) Vector(cols_n, ElemType) {
            var vec_out: Vector(rows_n, ElemType) = undefined;
            var sum: ElemType = 0;

            for (0..rows_n) |rr| {
                sum = 0;
                for (0..cols_n) |cc| {
                    sum += self.get(rr, cc) * vec.get(cc);
                }
                vec_out.elems[rr] = sum;
            }

            return vec_out;
        }

        pub fn multMat(self: *const Self, to_mult: Self) Self {
            var mat_out: Self = undefined;
            var sum: ElemType = 0;

            for (0..rows_n) |rr| {
                for (0..cols_n) |cc| {
                    sum = 0;

                    for (0..cols_n) |mm| {
                        sum += self.get(rr, mm) * to_mult.get(mm, cc);
                    }

                    mat_out.set(rr, cc, sum);
                }
            }
            return mat_out;
        }

        pub fn matPrint(self: *const Self) void {
            var ind: usize = 0;

            for (0..rows_n) |ii| {
                print("[", .{});
                for (0..cols_n) |jj| {
                    ind = (ii * cols_n) + jj;
                    print("{e:.3},", .{self.elems[ind]});
                }
                print("]\n", .{});
            }
            print("\n", .{});
        }
    };
}

pub fn Mat22T(comptime ElemType: type) type {
    return Matrix(2, 2, ElemType);
}

pub fn Mat33T(comptime ElemType: type) type {
    return Matrix(3, 3, ElemType);
}

pub fn Mat44T(comptime ElemType: type) type {
    return Matrix(4, 4, ElemType);
}

pub const Mat22Ops = struct {
    pub fn adj(ElemType: type, mat22: Mat22T(ElemType)) Mat22T(ElemType) {
        var adjoint: Mat22T(ElemType) = undefined;
        adjoint.set(0, 0, mat22.get(1, 1));
        adjoint.set(1, 1, mat22.get(0, 0));
        adjoint.set(0, 1, -1 * mat22.get(0, 1));
        adjoint.set(1, 0, -1 * mat22.get(1, 0));
        return adjoint;
    }

    pub fn det(ElemType: type, mat22: Mat22T(ElemType)) ElemType {
        return (mat22.get(0, 0) * mat22.get(1, 1)) - (mat22.get(0, 1) * mat22.get(1, 0));
    }

    pub fn inv(ElemType: type, mat22: Mat22T(ElemType)) Mat22T(ElemType) {
        var inverse: Mat22T(ElemType) = Mat22Ops.adj(ElemType, mat22);
        const determinant: ElemType = Mat22Ops.det(ElemType, mat22);
        inverse = inverse.multScalar(1 / determinant);
        return inverse;
    }
};

pub const Mat33Ops = struct {
    pub fn det(ElemType: type, mat33: Mat33T(ElemType)) ElemType {
        var sub_dets: [3]ElemType = undefined;

        sub_dets[0] = mat33.get(0, 0) * ((mat33.get(1, 1) * mat33.get(2, 2)) - (mat33.get(1, 2) * mat33.get(2, 1)));
        sub_dets[1] = mat33.get(0, 1) * ((mat33.get(1, 0) * mat33.get(2, 2)) - (mat33.get(1, 2) * mat33.get(2, 0)));
        sub_dets[2] = mat33.get(0, 2) * ((mat33.get(1, 0) * mat33.get(2, 1)) - (mat33.get(1, 1) * mat33.get(2, 0)));

        return sub_dets[0] - sub_dets[1] + sub_dets[2];
    }

    pub fn inv(ElemType: type, mat33: Mat33T(ElemType)) Mat33T(ElemType) {
        var inv33: Mat33T(ElemType) = undefined;

        const detm = 1 / Mat33Ops.det(ElemType, mat33);

        // Calculate the cofactors and transpose in one step
        inv33.elems[0] = detm * (mat33.get(1, 1) * mat33.get(2, 2) - mat33.get(1, 2) * mat33.get(2, 1));
        inv33.elems[1] = -detm * (mat33.get(0, 1) * mat33.get(2, 2) - mat33.get(0, 2) * mat33.get(2, 1));
        inv33.elems[2] = detm * (mat33.get(0, 1) * mat33.get(1, 2) - mat33.get(0, 2) * mat33.get(1, 1));
        inv33.elems[3] = -detm * (mat33.get(1, 0) * mat33.get(2, 2) - mat33.get(1, 2) * mat33.get(2, 0));
        inv33.elems[4] = detm * (mat33.get(0, 0) * mat33.get(2, 2) - mat33.get(0, 2) * mat33.get(2, 0));
        inv33.elems[5] = -detm * (mat33.get(0, 0) * mat33.get(1, 2) - mat33.get(0, 2) * mat33.get(1, 0));
        inv33.elems[6] = detm * (mat33.get(1, 0) * mat33.get(2, 1) - mat33.get(1, 1) * mat33.get(2, 0));
        inv33.elems[7] = -detm * (mat33.get(0, 0) * mat33.get(2, 1) - mat33.get(0, 1) * mat33.get(2, 0));
        inv33.elems[8] = detm * (mat33.get(0, 0) * mat33.get(1, 1) - mat33.get(0, 1) * mat33.get(1, 0));

        return inv33;
    }
};

const Mat44Ops = struct {
    // pub fn det(ElemType: type, mat: Mat44T(ElemType)) ElemType {


    // }
};

test "Mat22f.getRowVec" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const v0 = [_]EType{3,4};
    const vec_exp = Vec2f.initSlice(&v0);

    try expectEqual(vec_exp, mat0.getRowVec(1));
}

test "Mat22f.getColVec" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const v0 = [_]EType{1,3};
    const vec_exp = Vec2f.initSlice(&v0);

    try expectEqual(vec_exp, mat0.getColVec(0));
}

test "Mat22f.add" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const m1 = [_]EType{ 5, 6, 7, 8 };
    const mat1 = Mat22f.initSlice(&m1);

    const m2 = [_]EType{ 6, 8, 10, 12 };
    const mat_exp = Mat22f.initSlice(&m2);

    try expectEqual(mat0.add(mat1), mat_exp);
}

test "Mat22f.subtract" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const m1 = [_]EType{ 5, 6, 7, 8 };
    const mat1 = Mat22f.initSlice(&m1);

    const m2 = [_]EType{ -4, -4, -4, -4 };
    const mat_exp = Mat22f.initSlice(&m2);

    try expectEqual(mat_exp, mat0.subtract(mat1));
}

test "Mat22f.trace" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const trace_exp: EType = 5;

    try expectEqual(trace_exp, mat0.trace());
}

test "Mat22f.transpose" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const m1 = [_]EType{ 1, 3, 2, 4 };
    const mat_exp = Mat22f.initSlice(&m1);

    try expectEqual(mat_exp, mat0.transpose());
}

test "Mat22f.multScalar" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const scalar: EType = 2;
    const m1 = [_]EType{ 2, 4, 6, 8 };
    const mat_exp = Mat22f.initSlice(&m1);

    try expectEqual(mat_exp, mat0.multScalar(scalar));
}

test "Mat22f.multVec" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const v0 = [_]EType{ 1, 2 };
    const vec0 = Vec2f.initSlice(&v0);

    const v1 = [_]EType{ 5, 11 };
    const vec_exp = Vec2f.initSlice(&v1);

    try expectEqual(vec_exp, mat0.multVec(vec0));
}

test "Mat22f.multMat" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const m1 = [_]EType{ 4, 3, 2, 1 };
    const mat1 = Mat22f.initSlice(&m1);

    const m2 = [_]EType{ 8, 5, 20, 13 };
    const mat_exp = Mat22f.initSlice(&m2);

    try expectEqual(mat_exp, mat0.multMat(mat1));
}

test "Mat22Ops.adj" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const m2 = [_]EType{ 4, -2, -3, 1 };
    const mat_exp = Mat22f.initSlice(&m2);

    try expectEqual(mat_exp, Mat22Ops.adj(EType, mat0));
}

test "Mat22Ops.det" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const det_exp: EType = -2;

    try expectEqual(det_exp, Mat22Ops.det(EType, mat0));
}

test "Mat22Ops.inv" {
    const m0 = [_]EType{ 1, 2, 3, 4 };
    const mat0 = Mat22f.initSlice(&m0);

    const m2 = [_]EType{ -2, 1, 1.5, -0.5 };
    const mat_exp = Mat22f.initSlice(&m2);

    try expectEqual(mat_exp, Mat22Ops.inv(EType, mat0));
}

test "Mat33f.add" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);
    const mat1 = Mat33f.initSlice(&m0);

    const m2 = [_]EType{ 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    const mat_exp = Mat33f.initSlice(&m2);

    try expectEqual(mat_exp, mat0.add(mat1));
}

test "Mat33f.subtract" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);
    const mat1 = Mat33f.initSlice(&m0);

    const mat_exp = Mat33f.initZeros();

    try expectEqual(mat_exp, mat0.subtract(mat1));
}

test "Mat33f.getRowVec" {

}

test "Mat33f.getColVec" {

}

test "Mat33f.getSubMat" {

}

test "Mat33f.transpose" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);

    const m1 = [_]EType{ 1, 4, 7, 2, 5, 8, 3, 6, 9 };
    const mat_exp = Mat33f.initSlice(&m1);

    try expectEqual(mat_exp, mat0.transpose());
}

test "Mat33f.multScalar" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);

    const scalar: EType = 2;

    const m1 = [_]EType{ 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    const mat_exp = Mat33f.initSlice(&m1);

    try expectEqual(mat_exp, mat0.multScalar(scalar));
}

test "Mat33f.multVec" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);

    const v0 = [_]EType{ 3, 2, 1 };
    const vec0 = Vec3f.initSlice(&v0);

    const v1 = [_]EType{ 10, 28, 46 };
    const vec_exp = Vec3f.initSlice(&v1);

    try expectEqual(vec_exp, mat0.multVec(vec0));
}

test "Mat33f.multMat" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);

    const m1 = [_]EType{ 3, 1, 1, 1, 3, 1, 1, 1, 3 };
    const mat1 = Mat33f.initSlice(&m1);

    const m2 = [_]EType{ 8, 10, 12, 23, 25, 27, 38, 40, 42 };
    const mat_exp = Mat33f.initSlice(&m2);

    try expectEqual(mat_exp, mat0.multMat(mat1));
}

test "Mat33Ops.det" {
    const m0 = [_]EType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = Mat33f.initSlice(&m0);
    const det0_exp: EType = 0;

    const m1 = [_]EType{ 3, 1, 1, 1, 3, 1, 1, 1, 3 };
    const mat1 = Mat33f.initSlice(&m1);
    const det1_exp: EType = 20;

    try expectEqual(det0_exp, Mat33Ops.det(EType, mat0));
    try expectEqual(det1_exp, Mat33Ops.det(EType, mat1));
}

test "Mat33Ops.inv" {
    const m1 = [_]EType{ 3, 1, 1, 1, 3, 1, 1, 1, 3 };
    const mat1 = Mat33f.initSlice(&m1);

    const m2 = [_]EType{ 0.4, -0.1, -0.1, -0.1, 0.4, -0.1, -0.1, -0.1, 0.4 };
    const mat_exp = Mat33f.initSlice(&m2);

    try expectEqual(mat_exp, Mat33Ops.inv(EType,mat1));
}
