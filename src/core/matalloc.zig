const std = @import("std");
const print = std.debug.print;

const VecAlloc = @import("vecalloc.zig");

pub fn MatAlloc(comptime ElemType: type) type {
    return extern struct {
        elems: []ElemType,
        rows_n: usize,
        cols_n: usize,
        alloc: std.mem.Allocator,

        const Self: type = @This();

        pub fn init(allocator: std.mem.Allocator, rows_n: usize, cols_n: usize) !Self {
            return .{
                .elems = try allocator.alloc(ElemType, rows_n * cols_n),
                .rows_n = rows_n,
                .cols_n = cols_n,
                .alloc = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.alloc.free(self.elems);
        }

        pub fn fill(self: *const Self, fill_val: ElemType) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] = fill_val;
            }
        }

        pub fn fillDiag(self: *const Self, fill_val: ElemType, diag_val: ElemType) void {
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

        pub fn get(self: *const Self, row: usize, col: usize) ElemType {
            return self.elems[(row * self.cols_n) + col];
        }

        pub fn set(self: *Self, row: usize, col: usize, val: ElemType) void {
            self.elems[(row * self.cols_n) + col] = val;
        }

        // pub fn getRowVec(self: *const Self, row: usize) Vector(cols_n, ElemType) {
        //     // TODO: make this bounds check?
        //     const start: usize = row * cols_n;
        //     const end: usize = start + cols_n;
        //     const row_slice: []const ElemType = self.elems[start..end];
        //     const vec = Vector(cols_n, ElemType).initSlice(row_slice);
        //     return vec;
        // }

        // pub fn getColVec(self: *const Self, col: usize) Vector(rows_n, ElemType) {
        //     // TODO: make this bounds check?
        //     var col_vec: [rows_n]ElemType = undefined;
        //     for (0..rows_n) |rr| {
        //         col_vec[rr] = self.get(rr, col);
        //     }
        //     const vec = Vector(rows_n, ElemType).initSlice(&col_vec);
        //     return vec;
        // }

        // pub fn getSubMat(self: *const Self, row_start: usize, col_start: usize, comptime rows: usize, comptime cols: usize) Matrix(rows, cols, ElemType) {
        //     // TODO: make this bounds check?
        //     var sub_mat = Matrix(rows, cols, ElemType).initZeros();

        //     const row_end: usize = row_start + rows;
        //     const col_end: usize = col_start + cols;
        //     for (row_start..row_end) |rr| {
        //         for (col_start..col_end) |cc| {
        //             sub_mat.set(rr - row_start, cc - col_start, self.get(rr, cc));
        //         }
        //     }

        //     return sub_mat;
        // }

        // pub fn insertRowVec(self: *Self, row: usize, col_start: usize, comptime vec_len: usize, vec: Vector(vec_len, ElemType)) void {
        //     for (0..vec_len) |cc| {
        //         self.set(row, cc + col_start, vec.get(cc));
        //     }
        // }

        // pub fn insertColVec(self: *Self, col: usize, row_start: usize, comptime vec_len: usize, vec: Vector(vec_len, ElemType)) void {
        //     for (0..vec_len) |rr| {
        //         self.set(rr + row_start, col, vec.get(rr));
        //     }
        // }

        // pub fn insertSubMat(self: *Self, row_start: usize, col_start: usize, comptime mat_rows: usize, comptime mat_cols: usize, mat: Matrix(mat_rows, mat_rows, ElemType)) void {
        //     for (0..mat_rows) |rr| {
        //         for (0..mat_cols) |cc| {
        //             self.set(rr + row_start, cc + col_start, mat.get(rr, cc));
        //         }
        //     }
        // }

        pub fn transpose(self: *const Self) void {
            const temp_mat = try MatAlloc(ElemType).init(self.alloc, self.rows_n, self.cols_n);
            defer temp_mat.deinit();

            @memcpy(temp_mat.elems, self.elems);

            for (0..self.rows_n) |ii| {
                for (ii..self.cols_n) |jj| {
                    self.set(ii, jj, temp_mat.get(jj, ii));
                    self.set(jj, ii, temp_mat.get(ii, jj));
                }
            }
        }

        pub fn trace(self: *const Self) ElemType {
            var trace_out: ElemType = 0;

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

        pub fn mulScalar(self: *const Self, scalar: ElemType) Self {
            for (0..self.elems.len) |ee| {
                self.elems[ee] = scalar * self.elems[ee];
            }
        }

        // pub fn mulVec(self: *const Self, vec: Vector(cols_n, ElemType)) Vector(cols_n, ElemType) {
        //     var vec_out: Vector(rows_n, ElemType) = undefined;
        //     var sum: ElemType = 0;

        //     for (0..rows_n) |rr| {
        //         sum = 0;
        //         for (0..cols_n) |cc| {
        //             sum += self.get(rr, cc) * vec.get(cc);
        //         }
        //         vec_out.set(rr, sum);
        //     }

        //     return vec_out;
        // }

        // pub fn mulMat(self: *const Self, to_mult: Self) Self {
        //     var mat_out: Self = undefined;
        //     var sum: ElemType = 0;

        //     for (0..rows_n) |rr| {
        //         for (0..cols_n) |cc| {
        //             sum = 0;

        //             for (0..cols_n) |mm| {
        //                 sum += self.get(rr, mm) * to_mult.get(mm, cc);
        //             }

        //             mat_out.set(rr, cc, sum);
        //         }
        //     }
        //     return mat_out;
        // }

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
    };
}
