const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;

const slice = @import("slicetools.zig");
const ValInd = slice.ValInd;



pub fn VectorHeap(comptime elem_n: comptime_int, comptime ElemType: type) type {
    return extern struct {
        elems: []ElemType,
        alloc: std.mem.Allocator,

        const Self: type = @This();

        pub fn initFill(fill_val: ElemType) Self {
            return .{ .elems = [_]ElemType{fill_val} ** elem_n };
        }

        pub fn initOnes() Self {
            return initFill(1);
        }

        pub fn initZeros() Self {
            return initFill(0);
        }

        pub fn initSlice(slice_in: []const ElemType) Self {
            return .{ .elems = slice_in[0..elem_n].* };
        }

        pub fn get(self: *const Self, ind: usize) ElemType {
            return self.elems[ind];
        }

        pub fn set(self: *Self, ind: usize, val: ElemType) void {
            self.elems[ind] = val;
        }

        pub fn add(self: *const Self, to_add: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.elems[ii] = self.elems[ii] + to_add.elems[ii];
            }
            return vec_out;
        }

        pub fn subtract(self: *const Self, to_sub: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.elems[ii] = self.elems[ii] - to_sub.elems[ii];
            }
            return vec_out;
        }

        pub fn mulScalar(self: *const Self, scalar: ElemType) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.elems[ii] = scalar * self.elems[ii];
            }
            return vec_out;
        }

        pub fn dot(self: *const Self, to_dot: Self) ElemType {
            var dot_prod: ElemType = 0;
            for (0..elem_n) |ii| {
                dot_prod += self.elems[ii] * to_dot.elems[ii];
            }
            return dot_prod;
        }

        pub fn norm(self: *const Self) ElemType {
            var norm_out: ElemType = 0;
            for (0..elem_n) |ii| {
                norm_out += self.elems[ii] * self.elems[ii];
            }
            return norm_out;
        }

        pub fn length(self: *const Self) ElemType {
            return @sqrt(self.norm());
        }

        pub fn max(self: *const Self) ValInd(ElemType) {
            return slice.max(ElemType, &self.elems);
        }

        pub fn min(self: *const Self) ValInd(ElemType) {
            return slice.min(ElemType, &self.elems);
        }

        pub fn sum(self: *const Self) ElemType {
            return slice.sum(ElemType, &self.elems);
        }

        pub fn mean(self: *const Self) ElemType {
            return slice.mean(ElemType, &self.elems);
        }

        pub fn apply(self: *const Self, func: *const fn(val: anytype) ElemType) Self {
            var applied: Self = undefined;
            for (self.elems,0..) |elem,ii| {
                applied.elems[ii] = func(elem);
            }
            return applied;
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..elem_n) |ii| {
                print("{e:.3},", .{self.elems[ii]});
            }
            print("]\n", .{});
        }
    };
}