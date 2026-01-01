const std = @import("std");
const print = std.debug.print;

const testing = std.testing;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

const VecSlice = @import("vecslice.zig").VecSlice;
const sliceops = @import("sliceops.zig");

pub fn MatSlice(comptime EType: type) type {
    return struct {
        elems: []EType,
        rows_n: usize,
        cols_n: usize,

        const Self: type = @This();

        pub fn init(elems: []EType, rows_n: usize, cols_n: usize) !Self {
            assert(elems.len == (rows_n * cols_n));

            return .{
                .elems = elems,
                .rows_n = rows_n,
                .cols_n = cols_n,
            };
        }

        pub fn fill(self: *const Self, fill_val: EType) void {
            @memset(self.elems[0..], fill_val);
        }

        pub fn fillDiag(self: *const Self, fill_val: EType, diag_val: EType) void {
            for (0..self.rows_n) |ii| {
                for (ii..self.cols_n) |jj| {
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

        pub fn get(self: *const Self, row: usize, col: usize) EType {
            return self.elems[(row * self.cols_n) + col];
        }

        pub fn set(self: *Self, row: usize, col: usize, val: EType) void {
            self.elems[(row * self.cols_n) + col] = val;
        }

        pub fn transpose(self: *Self, buffer: *Self) !void {
            assert(self.cols_n == buffer.cols_n);
            assert(self.rows_n == buffer.rows_n);

            @memcpy(buffer.elems, self.elems);

            for (0..self.rows_n) |ii| {
                for (ii..self.cols_n) |jj| {
                    self.set(ii, jj, buffer.get(jj, ii));
                    self.set(jj, ii, buffer.get(ii, jj));
                }
            }
        }

        pub fn trace(self: *const Self) EType {
            var trace_out: EType = 0;

            if (self.rows_n <= self.cols_n) {
                for (0..self.rows_n) |ii| {
                    trace_out += self.get(ii, ii);
                }
            } else {
                for (0..self.cols_n) |ii| {
                    trace_out += self.get(ii, ii);
                }
            }

            return trace_out;
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            for (0..self.elems.len) |ee| {
                self.elems[ee] += to_add.elems[ee];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.elems.len) |ee| {
                self.elems[ee] -= to_sub.elems[ee];
            }
        }

        pub fn mulInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.elems.len) |ee| {
                self.elems[ee] *= to_sub.elems[ee];
            }
        }

        pub fn divInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.elems.len) |ee| {
                self.elems[ee] /= to_sub.elems[ee];
            }
        }

        pub fn mulScalarInPlace(self: *const Self, scalar: EType) void {
            for (0..self.elems.len) |ee| {
                self.elems[ee] = scalar * self.elems[ee];
            }
        }

		pub fn getSlice(self: *const Self, row_to_slice: usize) ![]EType {
			assert(row_to_slice <= self.rows_n);

			const start_ind: usize = row_to_slice*self.cols_n; 
			const end_ind: usize = start_ind+self.cols_n;
			return self.elems[start_ind..end_ind];
        }

        pub fn matPrint(self: *const Self) void {
            var ind: usize = 0;

            for (0..self.rows_n) |ii| {
                print("[", .{});
                for (0..self.cols_n) |jj| {
                    ind = (ii * self.cols_n) + jj;
                    print("{e:.3},", .{self.elems[ind]});
                }
                print("]\n", .{});
            }
            print("\n", .{});
        }

        pub fn saveCSV(self: *const Self,
                       io: std.Io, 
                       out_dir: std.Io.Dir, 
                       file_name: []const u8) !void {
                       
            const csv_file = try out_dir.createFile(io, file_name, .{});
            defer csv_file.close(io);

            var write_buf: [4096]u8 = undefined;
            var file_writer = csv_file.writer(io,&write_buf);
            const writer = &file_writer.interface;

            for (0..self.rows_n) |rr| {
                for (0..self.cols_n) |cc| {
                    try writer.print("{d},", .{self.get(rr, cc)});
                }
                try writer.print("\n",.{});
            }
            try writer.flush();
        }
    };
}

pub fn MatSliceOps(comptime EType: type) type {
    return struct {
        pub fn add(mat0: *const MatSlice(EType), 
                   mat1: *const MatSlice(EType), 
                   mat_out: *MatSlice(EType)) !void {
            assert(mat0.rows_n == mat1.rows_n);
            assert(mat0.cols_n == mat1.cols_n);

            for (0..mat0.elems.len) |ii| {
                mat_out.elems[ii] = mat0.elems[ii] + mat1.elems[ii];
            }
        }

        pub fn sub(mat0: *const MatSlice(EType), 
                   mat1: *const MatSlice(EType), 
                   mat_out: *MatSlice(EType)) !void {
            assert(mat0.rows_n == mat1.rows_n);
            assert(mat0.cols_n == mat1.cols_n);

            for (0..mat0.elems.len) |ii| {
                mat_out.elems[ii] = mat0.elems[ii] - mat1.elems[ii];
            }
        }

        pub fn mulElemWise(mat0: *const MatSlice(EType), 
                           mat1: *const MatSlice(EType), 
                           mat_out: *MatSlice(EType)) !void {
            assert(mat0.rows_n == mat1.rows_n);
            assert(mat0.cols_n == mat1.cols_n);

            for (0..mat0.elems.len) |ii| {
                mat_out.elems[ii] = mat0.elems[ii] * mat1.elems[ii];
            }
        }

        pub fn divElemWise(mat0: *const MatSlice(EType), 
                           mat1: *const MatSlice(EType), 
                           mat_out: *MatSlice(EType)) !void {
            assert(mat0.rows_n == mat1.rows_n);
            assert(mat0.cols_n == mat1.cols_n);

            for (0..mat0.elems.len) |ii| {
                mat_out.elems[ii] = mat0.elems[ii] / mat1.elems[ii];
            }
        }

        pub fn mulScalar(mat0: *const MatSlice(EType), 
                         scalar: EType, 
                         mat_out: *MatSlice(EType)) !void {
            for (0..mat0.elems.len) |ii| {
                mat_out.elems[ii] = scalar * mat0.elems[ii];
            }
        }

        pub fn mulVec(mat: *const MatSlice(EType), 
                      vec_mul: *const VecSlice(EType), 
                      vec_out: *VecSlice(EType)) !void {
            assert(mat.cols_n == vec_mul.elems.len);

            var sum: EType = 0;

            for (0..mat.rows_n) |rr| {
                sum = 0;
                for (0..mat.cols_n) |cc| {
                    sum += mat.get(rr, cc) * vec_mul.get(cc);
                }
                vec_out.set(rr, sum);
            }
        }

        pub fn mulMat(mat0: *const MatSlice(EType), 
                      mat1: *const MatSlice(EType), 
                      mat_out: *MatSlice(EType)) !void {
                      
            assert(mat0.cols_n == mat1.rows_n);

            var sum: EType = 0;

            for (0..mat0.rows_n) |rr| {
                for (0..mat0.cols_n) |cc| {
                    sum = 0;

                    for (0..mat1.cols_n) |mm| {
                        sum += mat0.get(rr, mm) * mat1.get(mm, cc);
                    }

                    mat_out.set(rr, cc, sum);
                }
            }
        }
    };
}

//TODO: transfer missing tests from stack matrix
const TestType = f64;
const talloc = testing.allocator;

test "MatSlice.getSlice" {
	const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    var mat0 = try MatSlice(TestType).init(m0, rows, cols);
    mat0.fill(0.0);

    for (0..cols) |cc| {
    	mat0.set(1,cc,7);
    }
    for (0..cols) |cc| {
       	mat0.set(2,cc,9);
    }

    const exp0 = [_]TestType{0} ** 4;
    const exp1 = [_]TestType{7} ** 4;
    const exp2 = [_]TestType{9} ** 4;

	const slice0 = try mat0.getSlice(0);
	const slice1 = try mat0.getSlice(1);
	const slice2 = try mat0.getSlice(2);

    // mat0.matPrint();
    // print("exp0={any}\n",.{exp0});
    // print("slice0={any}\n",.{slice0});
    // print("exp1={any}\n",.{exp1});
    // print("slice1={any}\n",.{slice1});
    // print("exp2={any}\n",.{exp2});
    // print("slice2={any}\n",.{slice2});

    try expectEqualSlices(TestType, exp0[0..], slice0);
    try expectEqualSlices(TestType, exp1[0..], slice1);
    try expectEqualSlices(TestType, exp2[0..], slice2);
}

test "MatSliceOps.add" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = try MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = try MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = try MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(2.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = try MatSlice(TestType).init(m_op, rows, cols);

    try MatSliceOps(TestType).add(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.elems, mat_op.elems);
}

test "MatSliceOps.sub" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = try MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = try MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = try MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(0.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = try MatSlice(TestType).init(m_op, rows, cols);

    try MatSliceOps(TestType).sub(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.elems, mat_op.elems);
}

test "MatSliceOps.mulElemWise" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = try MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = try MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = try MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(1.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = try MatSlice(TestType).init(m_op, rows, cols);

    try MatSliceOps(TestType).mulElemWise(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.elems, mat_op.elems);
}

test "MatSliceOps.mulScalar" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = try MatSlice(TestType).init(m0, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = try MatSlice(TestType).init(m_exp, rows, cols);

    const scalar: TestType = 2.0;

    mat0.fill(1.0);
    mat_exp.fill(2.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = try MatSlice(TestType).init(m_op, rows, cols);

    try MatSliceOps(TestType).mulScalar(&mat0, scalar, &mat_op);

    try expectEqualSlices(TestType, mat_exp.elems, mat_op.elems);
}

test "MatSliceOps.divElemWise" {
    const rows: usize = 3;
    const cols: usize = 4;

    const m0 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m0);
    const mat0 = try MatSlice(TestType).init(m0, rows, cols);

    const m1 = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m1);
    const mat1 = try MatSlice(TestType).init(m1, rows, cols);

    const m_exp = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_exp);
    const mat_exp = try MatSlice(TestType).init(m_exp, rows, cols);

    mat0.fill(1.0);
    mat1.fill(1.0);
    mat_exp.fill(1.0);

    const m_op = try talloc.alloc(TestType, rows * cols);
    defer talloc.free(m_op);
    var mat_op = try MatSlice(TestType).init(m_op, rows, cols);

    try MatSliceOps(TestType).divElemWise(&mat0, &mat1, &mat_op);

    try expectEqualSlices(TestType, mat_exp.elems, mat_op.elems);
}

test "MatSlice.addInPlace" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = try MatSlice(TestType).init(m0[0..], 2, 2);

    var m1 = [_]TestType{ 5, 6, 7, 8 };
    const mat1 = try MatSlice(TestType).init(m1[0..], 2, 2);

    var m2 = [_]TestType{ 6, 8, 10, 12 };
    const mat_exp = try MatSlice(TestType).init(m2[0..], 2, 2);

    mat0.addInPlace(&mat1);

    try expectEqualSlices(TestType, mat_exp.elems, mat0.elems);
}

test "MatSlice.subInPlace" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = try MatSlice(TestType).init(m0[0..], 2, 2);

    var m1 = [_]TestType{ 5, 6, 7, 8 };
    const mat1 = try MatSlice(TestType).init(m1[0..], 2, 2);

    var m2 = [_]TestType{ -4, -4, -4, -4 };
    const mat_exp = try MatSlice(TestType).init(m2[0..], 2, 2);

    mat0.subInPlace(&mat1);

    try expectEqualSlices(TestType, mat_exp.elems, mat0.elems);
}

test "MatSlice.trace" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = try MatSlice(TestType).init(m0[0..], 2, 2);

    const trace_exp: TestType = 5;

    try expectEqual(trace_exp, mat0.trace());
}

test "MatSlice.transpose" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    var mat0 = try MatSlice(TestType).init(m0[0..], 2, 2);

    var m_buff = [_]TestType{ 0, 0, 0, 0 };
    var mat_buff = try MatSlice(TestType).init(m_buff[0..], 2, 2);

    var m1 = [_]TestType{ 1, 3, 2, 4 };
    const mat_exp = try MatSlice(TestType).init(m1[0..], 2, 2);

    try mat0.transpose(&mat_buff);

    try expectEqualSlices(TestType, mat_exp.elems, mat0.elems);
}

test "MatSlice.mulScalar" {
    var m0 = [_]TestType{ 1, 2, 3, 4 };
    const mat0 = try MatSlice(TestType).init(m0[0..], 2, 2);

    const scalar: TestType = 2;

    var m1 = [_]TestType{ 2, 4, 6, 8 };
    const mat_exp = try MatSlice(TestType).init(m1[0..], 2, 2);

    mat0.mulScalarInPlace(scalar);

    try expectEqualSlices(TestType, mat_exp.elems, mat0.elems);
}

test "MatSliceOps.mulVec" {
    var m0 = [_]TestType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = try MatSlice(TestType).init(&m0, 3, 3);

    var v0 = [_]TestType{ 3, 2, 1 };
    const vec0 = VecSlice(TestType).init(&v0);

    var v1 = [_]TestType{ 10, 28, 46 };
    const vec_exp = VecSlice(TestType).init(&v1);

    var v_out = [_]TestType{0} ** 3;
    var vec_out = VecSlice(TestType).init(&v_out);

    try MatSliceOps(TestType).mulVec(&mat0, &vec0, &vec_out);

    try expectEqualSlices(TestType, vec_exp.elems, vec_out.elems);
}

test "MatSliceOps.mulMat" {
    var m0 = [_]TestType{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat0 = try MatSlice(TestType).init(&m0, 3, 3);

    var m1 = [_]TestType{ 3, 1, 1, 1, 3, 1, 1, 1, 3 };
    const mat1 = try MatSlice(TestType).init(&m1, 3, 3);

    var m2 = [_]TestType{0} ** 9;
    var mat_out = try MatSlice(TestType).init(&m2, 3, 3);

    var m3 = [_]TestType{ 8, 10, 12, 23, 25, 27, 38, 40, 42 };
    const mat_exp = try MatSlice(TestType).init(&m3, 3, 3);

    try MatSliceOps(TestType).mulMat(&mat0, &mat1, &mat_out);

    try expectEqualSlices(TestType, mat_exp.elems, mat_out.elems);
}

