// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const print = std.debug.print;
const assert = std.debug.assert;

const VecSlice = @import("vecslice.zig").VecSlice;
const sliceops = @import("sliceops.zig");
const csvio = @import("csvio.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub fn MatSlice(comptime T: type) type {
    return struct {
        slice: []T,
        rows_num: usize,
        cols_num: usize,

        const Self: type = @This();

        pub fn init(slice: []T, rows_num: usize, cols_num: usize) Self {
            assert(slice.len == (rows_num * cols_num));

            return .{
                .slice = slice,
                .rows_num = rows_num,
                .cols_num = cols_num,
            };
        }

        pub fn initAlloc(
            outer_alloc: std.mem.Allocator,
            rows_num: usize,
            cols_num: usize,
        ) !Self {
            const slice = try outer_alloc.alloc(T, rows_num * cols_num);

            return init(slice, rows_num, cols_num);
        }

        pub fn fill(self: *const Self, fill_val: T) void {
            @memset(self.slice[0..], fill_val);
        }

        pub fn fillDiag(self: *const Self, fill_val: T, diag_val: T) void {
            for (0..self.rows_num) |ii| {
                for (ii..self.cols_num) |jj| {
                    if (ii == jj) {
                        self.set(ii, jj, diag_val);
                    } else {
                        self.set(ii, jj, fill_val);
                    }
                }
            }
        }

        pub fn identity(self: *const Self) void {
            self.fillDiag(0, 1);
        }

        pub fn get(self: *const Self, row: usize, col: usize) T {
            assert(row < self.rows_num);
            assert(col < self.cols_num);
            return self.slice[(row * self.cols_num) + col];
        }

        pub fn set(self: *Self, row: usize, col: usize, val: T) void {
            assert(row < self.rows_num);
            assert(col < self.cols_num);
            self.slice[(row * self.cols_num) + col] = val;
        }

        pub fn flatIndex(self: *const Self, row: usize, col: usize) usize {
            assert(row < self.rows_num);
            assert(col < self.cols_num);
            return row * self.cols_num + col;
        }

        pub fn rowBase(self: *const Self, row: usize) usize {
            assert(row < self.rows_num);
            return row * self.cols_num;
        }

        pub fn getFlat(self: *const Self, flat_idx: usize) T {
            assert(flat_idx < self.slice.len);
            return self.slice[flat_idx];
        }

        pub fn setFlat(self: *Self, flat_idx: usize, val: T) void {
            assert(flat_idx < self.slice.len);
            self.slice[flat_idx] = val;
        }

        pub fn transpose(self: *Self, buff: *Self) void {
            assert(self.cols_num == buff.cols_num);
            assert(self.rows_num == buff.rows_num);

            @memcpy(buff.slice, self.slice);

            for (0..self.rows_num) |ii| {
                for (ii..self.cols_num) |jj| {
                    self.set(ii, jj, buff.get(jj, ii));
                    self.set(jj, ii, buff.get(ii, jj));
                }
            }
        }

        pub fn trace(self: *const Self) T {
            var trace_out: T = 0;

            if (self.rows_num <= self.cols_num) {
                for (0..self.rows_num) |ii| {
                    trace_out += self.get(ii, ii);
                }
            } else {
                for (0..self.cols_num) |ii| {
                    trace_out += self.get(ii, ii);
                }
            }

            return trace_out;
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            for (0..self.slice.len) |ee| {
                self.slice[ee] += to_add.slice[ee];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.slice.len) |ee| {
                self.slice[ee] -= to_sub.slice[ee];
            }
        }

        pub fn mulInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.slice.len) |ee| {
                self.slice[ee] *= to_sub.slice[ee];
            }
        }

        pub fn divInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.slice.len) |ee| {
                self.slice[ee] /= to_sub.slice[ee];
            }
        }

        pub fn mulScalInPlace(self: *const Self, scal: T) void {
            for (0..self.slice.len) |ee| {
                self.slice[ee] = scal * self.slice[ee];
            }
        }

        pub fn getSlice(self: *const Self, row_to_slice: usize) []T {
            assert(row_to_slice < self.rows_num);

            const start_idx: usize = row_to_slice * self.cols_num;
            const end_idx: usize = start_idx + self.cols_num;
            return self.slice[start_idx..end_idx];
        }

        pub fn matPrint(self: *const Self) void {
            var ind: usize = 0;

            for (0..self.rows_num) |ii| {
                print("[", .{});
                for (0..self.cols_num) |jj| {
                    ind = (ii * self.cols_num) + jj;
                    print("{e:.3},", .{self.slice[ind]});
                }
                print("]\n", .{});
            }
            print("\n", .{});
        }

        pub fn minByRow(self: *const Self, fixed_col: usize) T {
            assert(fixed_col < self.cols_num);

            var val: T = self.get(0, fixed_col);

            for (1..self.rows_num) |ii| {
                const check = self.get(ii, fixed_col);
                if (check < val) {
                    val = check;
                }
            }

            return val;
        }

        pub fn maxByRow(self: *const Self, fixed_col: usize) T {
            assert(fixed_col < self.cols_num);

            var val: T = self.get(0, fixed_col);

            for (1..self.rows_num) |ii| {
                const check = self.get(ii, fixed_col);
                if (check > val) {
                    val = check;
                }
            }

            return val;
        }

        pub fn saveCSV(
            self: *const Self,
            io: std.Io,
            out_dir: std.Io.Dir,
            file_name: []const u8,
        ) !void {
            const SaveCtx = struct {
                fn getVal(ctx: *const Self, row: usize, col: usize) T {
                    return ctx.get(row, col);
                }
            };

            try csvio.saveScalarGridCSV(
                io,
                out_dir,
                file_name,
                self.rows_num,
                self.cols_num,
                self,
                SaveCtx.getVal,
            );
        }
    };
}

pub fn MatSliceOps(comptime T: type) type {
    return struct {
        pub fn add(
            mat0: *const MatSlice(T),
            mat1: *const MatSlice(T),
            mat_out: *MatSlice(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] + mat1.slice[ii];
            }
        }

        pub fn sub(
            mat0: *const MatSlice(T),
            mat1: *const MatSlice(T),
            mat_out: *MatSlice(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] - mat1.slice[ii];
            }
        }

        pub fn mulElemWise(
            mat0: *const MatSlice(T),
            mat1: *const MatSlice(T),
            mat_out: *MatSlice(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] * mat1.slice[ii];
            }
        }

        pub fn divElemWise(
            mat0: *const MatSlice(T),
            mat1: *const MatSlice(T),
            mat_out: *MatSlice(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] / mat1.slice[ii];
            }
        }

        pub fn mulScal(
            mat0: *const MatSlice(T),
            scal: T,
            mat_out: *MatSlice(T),
        ) void {
            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = scal * mat0.slice[ii];
            }
        }

        pub fn mulVec(
            mat: *const MatSlice(T),
            vec_mul: *const VecSlice(T),
            vec_out: *VecSlice(T),
        ) void {
            assert(mat.cols_num == vec_mul.slice.len);

            var sum: T = 0;

            for (0..mat.rows_num) |rr| {
                sum = 0;
                for (0..mat.cols_num) |cc| {
                    sum += mat.get(rr, cc) * vec_mul.get(cc);
                }
                vec_out.set(rr, sum);
            }
        }

        pub fn mulMat(
            mat0: *const MatSlice(T),
            mat1: *const MatSlice(T),
            mat_out: *MatSlice(T),
        ) void {
            assert(mat0.cols_num == mat1.rows_num);

            var sum: T = 0;

            for (0..mat0.rows_num) |rr| {
                for (0..mat0.cols_num) |cc| {
                    sum = 0;

                    for (0..mat1.cols_num) |mm| {
                        sum += mat0.get(rr, mm) * mat1.get(mm, cc);
                    }

                    mat_out.set(rr, cc, sum);
                }
            }
        }
    };
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const testing = std.testing;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;
const TestType = F;
const talloc = testing.allocator;

test "MatSlice.getSlice" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    var mat0 = MatSlice(TestType).init(m0, rows, cols);
    mat0.fill(0.0);

    for (0..cols) |cc| {
        mat0.set(1, cc, 7);
    }
    for (0..cols) |cc| {
        mat0.set(2, cc, 9);
    }

    const exp0 = [_]TestType{0} ** 4;
    const exp1 = [_]TestType{7} ** 4;
    const exp2 = [_]TestType{9} ** 4;

    const slice0 = mat0.getSlice(0);
    const slice1 = mat0.getSlice(1);
    const slice2 = mat0.getSlice(2);

    try expectEqualSlices(TestType, exp0[0..], slice0);
    try expectEqualSlices(TestType, exp1[0..], slice1);
    try expectEqualSlices(TestType, exp2[0..], slice2);
}

test "MatSlice flat helpers" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    var mat0 = MatSlice(TestType).init(m0, rows, cols);
    mat0.fill(0.0);

    const base = mat0.rowBase(2);
    const idx = mat0.flatIndex(2, 3);
    mat0.setFlat(idx, 9.0);

    try expectEqual(@as(usize, 8), base);
    try expectEqual(@as(usize, 11), idx);
    try expectApproxEqAbs(@as(TestType, 9.0), mat0.get(2, 3), 1e-12);
}

test "MatSliceOps.add" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(2.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = MatSlice(TestType).init(m_op, rows, cols);

    MatSliceOps(TestType).add(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.slice, mat_op.slice);
}

test "MatSliceOps.sub" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(0.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = MatSlice(TestType).init(m_op, rows, cols);

    MatSliceOps(TestType).sub(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.slice, mat_op.slice);
}

test "MatSliceOps.mulElemWise" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(1.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = MatSlice(TestType).init(m_op, rows, cols);

    MatSliceOps(TestType).mulElemWise(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.slice, mat_op.slice);
}

test "MatSliceOps.mulScal" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = MatSlice(TestType).init(m0, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = MatSlice(TestType).init(m_exp, rows, cols);

    const scal: TestType = 2.0;

    mat0.fill(1.0);
    mat_exp.fill(2.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = MatSlice(TestType).init(m_op, rows, cols);

    MatSliceOps(TestType).mulScal(&mat0, scal, &mat_op);

    try expectEqualSlices(TestType, mat_exp.slice, mat_op.slice);
}

test "MatSliceOps.divElemWise" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(1.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = MatSlice(TestType).init(m_op, rows, cols);

    MatSliceOps(TestType).divElemWise(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.slice, mat_op.slice);
}

test "MatSlice.addInPlace" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = MatSlice(TestType).init(m0[0..], 2, 2);

    var m1 = [_]TestType{ 5, 6, 7, 8 };
    const mat1 = MatSlice(TestType).init(m1[0..], 2, 2);

    var m2 = [_]TestType{ 6, 8, 10, 12 };
    const mat_exp = MatSlice(TestType).init(m2[0..], 2, 2);

    mat0.addInPlace(&mat1);

    try expectEqualSlices(TestType, mat_exp.slice, mat0.slice);
}

test "MatSlice.subInPlace" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = MatSlice(TestType).init(m0[0..], 2, 2);

    var m1 = [_]TestType{ 5, 6, 7, 8 };
    const mat1 = MatSlice(TestType).init(m1[0..], 2, 2);

    var m2 = [_]TestType{ -4, -4, -4, -4 };
    const mat_exp = MatSlice(TestType).init(m2[0..], 2, 2);

    mat0.subInPlace(&mat1);

    try expectEqualSlices(TestType, mat_exp.slice, mat0.slice);
}

test "MatSlice.trace" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = MatSlice(TestType).init(m0[0..], 2, 2);

    const trace_exp: TestType = 5;

    try expectEqual(trace_exp, mat0.trace());
}

test "MatSlice.transpose" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    var mat0 = MatSlice(TestType).init(m0[0..], 2, 2);

    var m_buff = [_]TestType{ 0, 0, 0, 0 };
    var mat_buff = MatSlice(TestType).init(m_buff[0..], 2, 2);

    var m1 = [_]TestType{ 1, 3, 2, 4 };
    const mat_exp = MatSlice(TestType).init(m1[0..], 2, 2);

    mat0.transpose(&mat_buff);

    try expectEqualSlices(TestType, mat_exp.slice, mat0.slice);
}

test "MatSlice.mulScal" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = MatSlice(TestType).init(m0[0..], 2, 2);

    const scal: TestType = 2;

    var m1 = [_]TestType{ 2, 4, 6, 8 };
    const mat_exp = MatSlice(TestType).init(m1[0..], 2, 2);

    mat0.mulScalInPlace(scal);

    try expectEqualSlices(TestType, mat_exp.slice, mat0.slice);
}

test "MatSliceOps.mulVec" {
    var m0 = [_]TestType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = MatSlice(TestType).init(&m0, 3, 3);

    var v0 = [_]TestType{ 3, 2, 1 };
    const vec0 = VecSlice(TestType).init(&v0);

    var v1 = [_]TestType{ 10, 28, 46 };
    const vec_exp = VecSlice(TestType).init(&v1);

    var v_out = [_]TestType{0} ** 3;
    var vec_out = VecSlice(TestType).init(&v_out);

    MatSliceOps(TestType).mulVec(&mat0, &vec0, &vec_out);

    try expectEqualSlices(TestType, vec_exp.slice, vec_out.slice);
}

test "MatSliceOps.mulMat" {
    var m0 = [_]TestType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = MatSlice(TestType).init(&m0, 3, 3);

    var m1 = [_]TestType{ 3, 1, 1, 1, 3, 1, 1, 1, 3 };
    const mat1 = MatSlice(TestType).init(&m1, 3, 3);

    var m2 = [_]TestType{0} ** 9;
    var mat_out = MatSlice(TestType).init(&m2, 3, 3);

    var m3 = [_]TestType{ 8, 10, 12, 23, 25, 27, 38, 40, 42 };
    const mat_exp = MatSlice(TestType).init(&m3, 3, 3);

    MatSliceOps(TestType).mulMat(&mat0, &mat1, &mat_out);

    try expectEqualSlices(TestType, mat_exp.slice, mat_out.slice);
}
