const std = @import("std");
const print = std.debug.print;

const VecHeap = @import("vecheap.zig");

pub fn MatrixAlloc(comptime rows_n: comptime_int, comptime cols_n: comptime_int, comptime ElemType: type) type {
    return extern struct {
        elems: []ElemType,
        alloc: std.mem.Allocator,

        pub const elem_n: usize = rows_n * cols_n;

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

        pub fn initDiag(diag_val: ElemType) Self {
            var ident: Self = initZeros();

            var diag_n: usize = cols_n;
            if (cols_n > rows_n) {
                diag_n = rows_n;
            }

            for (0..diag_n) |ii| {
                ident.set(ii, ii, diag_val);
            }

            return ident;
        }

        pub fn initIdentity() Self {
            return initDiag(1);
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

        pub fn getRowVec(self: *const Self, row: usize) Vector(cols_n, ElemType) {
            // TODO: make this bounds check?
            const start: usize = row * cols_n;
            const end: usize = start + cols_n;
            const row_slice: []const ElemType = self.elems[start..end];
            const vec = Vector(cols_n, ElemType).initSlice(row_slice);
            return vec;
        }

        pub fn getColVec(self: *const Self, col: usize) Vector(rows_n, ElemType) {
            // TODO: make this bounds check?
            var col_vec: [rows_n]ElemType = undefined;
            for (0..rows_n) |rr| {
                col_vec[rr] = self.get(rr, col);
            }
            const vec = Vector(rows_n, ElemType).initSlice(&col_vec);
            return vec;
        }

        pub fn getSubMat(self: *const Self, row_start: usize, col_start: usize, comptime rows: usize, comptime cols: usize) Matrix(rows, cols, ElemType) {
            // TODO: make this bounds check?
            var sub_mat = Matrix(rows, cols, ElemType).initZeros();

            const row_end: usize = row_start + rows;
            const col_end: usize = col_start + cols;
            for (row_start..row_end) |rr| {
                for (col_start..col_end) |cc| {
                    sub_mat.set(rr - row_start, cc - col_start, self.get(rr, cc));
                }
            }

            return sub_mat;
        }

        pub fn insertRowVec(self: *Self, row: usize, col_start: usize, comptime vec_len: usize, vec: Vector(vec_len, ElemType)) void {
            for (0..vec_len) |cc| {
                self.set(row, cc + col_start, vec.get(cc));
            }
        }

        pub fn insertColVec(self: *Self, col: usize, row_start: usize, comptime vec_len: usize, vec: Vector(vec_len, ElemType)) void {
            for (0..vec_len) |rr| {
                self.set(rr + row_start, col, vec.get(rr));
            }
        }

        pub fn insertSubMat(self: *Self, row_start: usize, col_start: usize, comptime mat_rows: usize, comptime mat_cols: usize, mat: Matrix(mat_rows, mat_rows, ElemType)) void {
            for (0..mat_rows) |rr| {
                for (0..mat_cols) |cc| {
                    self.set(rr + row_start, cc + col_start, mat.get(rr, cc));
                }
            }
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
                trace_out += self.get(ii, ii);
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
                vec_out.set(rr, sum);
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
