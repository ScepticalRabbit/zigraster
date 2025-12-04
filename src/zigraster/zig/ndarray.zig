const std = @import("std");
const print = std.debug.print;

const testing = std.testing;
const assert = std.debug.assert;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

const MatSlice = @import("matslice.zig").MatSlice;
const sliceops = @import("sliceops.zig");

const NDArrayError = error{
    ElemsWrongLenForDims,
    IndexOutOfBounds,
    DimOutOfBounds,
    IndicesWrongLenForDims,
    DimMismatch,
};

pub fn NDArray(comptime EType: type) type {
    return struct {
        elems: []EType,
        dims: []usize,
        strides: []usize,

        const Self: type = @This();

        pub fn init(allocator: std.mem.Allocator, 
        			elems: []EType, 
        			dims: []usize) !Self {
        			
            var dim_prod: usize = dims[0];
            for (1..dims.len) |dd| {
                dim_prod *= dims[dd];
            }

            if (elems.len != dim_prod) {
                return NDArrayError.ElemsWrongLenForDims;
            }

            const strides = try allocator.alloc(usize, dims.len);
            var ndarray = NDArray(EType){ .elems = elems, 
            							  .dims = dims, 
            							  .strides = strides };

            for (0..dims.len) |dd| {
                ndarray.strides[dd] = try ndarray.calcFlatStride(dd);
            }

            return ndarray;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.strides);
        }

        pub fn set(self: *Self, indices: []const usize, in_val: EType) !void {
            const ind: usize = try self.getFlatInd(indices);
            self.elems[ind] = in_val;
        }

        pub fn get(self: *Self, indices: []const usize) !EType {
            const ind: usize = try self.getFlatInd(indices);
            return self.elems[ind];
        }

        pub fn getFlatInd(self: *const Self, indices: []const usize) !usize {
            if (indices.len != self.dims.len) {
                return NDArrayError.IndicesWrongLenForDims;
            }

            var flat: usize = 0;

            for (indices, 0..) |ind, dim| {
                if (ind >= self.dims[dim]) {
                    return NDArrayError.IndexOutOfBounds;
                }

                flat += ind * self.strides[dim];
            }

            return flat;
        }

        pub fn calcFlatStride(self: *const Self, dim_ind: usize) !usize {
            // Row-major (C format) for flat NDArrays
            // stride = product of all dimensions to the right of the selected
            // dimension.
            if (dim_ind >= self.dims.len) {
                return NDArrayError.DimOutOfBounds;
            }

            var stride: usize = 1;

            var ii: usize = dim_ind + 1;
            while (ii < self.dims.len) : (ii += 1) {
                stride *= self.dims[ii];
            }

            return stride;
        }

        pub fn fill(self: *const Self, fill_val: EType) void {
            @memset(self.elems[0..], fill_val);
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            if (!matchArrayDims(EType, self, to_add)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] += to_add.elems[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            if (!matchArrayDims(EType, self, to_sub)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] -= to_sub.elems[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            if (!matchArrayDims(EType, self, to_mul)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] *= to_mul.elems[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            if (!matchArrayDims(EType, self, to_div)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] /= to_div.elems[ii];
            }
        }

        pub fn mulScalarInPlace(self: *const Self, scalar: EType) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] *= scalar;
            }
        }

        pub fn getEndDimSlice(self: *const Self,
							  fixed_inds: []usize,	
                              slice_fixed_dim: usize) ![]EType {
                              
            if (fixed_inds.len != self.dims.len) {
            	return NDArrayError.IndicesWrongLenForDims;
            }
                        
            if ((slice_fixed_dim+1) >= self.dims.len) {
            	return NDArrayError.DimOutOfBounds;
            } 

			// Have to zero off all other dimensions to ensure slice starts at
			// the correct location
			for (slice_fixed_dim+1..fixed_inds.len) |ii| {
				fixed_inds[ii] = 0;
			}
				
        	const start_ind: usize = try self.getFlatInd(fixed_inds);
        	const end_ind: usize = start_ind+self.strides[slice_fixed_dim];

	    	return self.elems[start_ind..end_ind];                     	
        } 
    };
}

pub fn NDArrayOps(comptime EType: type) type {
    return struct {
        pub fn add(arr0: *const NDArray(EType), 
                   arr1: *const NDArray(EType), 
                   arr_out: *NDArray(EType)) !void {

            if (!matchArrayDims(EType, arr0, arr1) 
                or !matchArrayDims(EType, arr0, arr_out)) {
                return NDArrayError.DimMismatch;
            }
            sliceops.add(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn sub(arr0: *const NDArray(EType), 
        	       arr1: *const NDArray(EType), 
        	       arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)) {
                return NDArrayError.DimMismatch;
            }
            sliceops.sub(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn mulElemWise(arr0: *const NDArray(EType), 
        				   arr1: *const NDArray(EType), 
        				   arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)) {
                return NDArrayError.DimMismatch;
            }
            sliceops.mul(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn divElemWise(arr0: *const NDArray(EType), 
        				  arr1: *const NDArray(EType), 
        				  arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)) {
                return NDArrayError.DimMismatch;
            }
            sliceops.div(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn extractMat(allocator: std.mem.Allocator,
                          arr: *const NDArray(EType), 
						  fixed_inds: []const usize,	
        				  row_ext: usize, 
        				  col_ext: usize,
        				  mat: *MatSlice(EType)) !void {
        				  
			if (fixed_inds.len != arr.dims.len) {
                return NDArrayError.IndicesWrongLenForDims;
            }
            
            if (row_ext > arr.dims.len or col_ext > arr.dims.len) {
            	return NDArrayError.DimOutOfBounds;
            }
            
            const num_elems: usize = arr.dims[row_ext]*arr.dims[col_ext];
            if (num_elems != mat.elems.len) {
            	return NDArrayError.ElemsWrongLenForDims; 	
            } 

            var get_dims_slice: []usize = try allocator.dupe(usize, fixed_inds);
			defer allocator.free(get_dims_slice);

            for (0..arr.dims[row_ext]) |rr| {
                for (0..arr.dims[col_ext]) |cc| {
                
					get_dims_slice[row_ext] = rr;
					get_dims_slice[col_ext] = cc;

                    mat.set(rr,cc, arr.get(get_dims_slice));
                }
            }
        }
    };
}

pub fn matchArrayDims(comptime EType: type, 
					  arr0: *const NDArray(EType), 
					  arr1: *const NDArray(EType)) bool {
					  
    if (arr0.dims.len != arr1.dims.len) {
        return false;
    }

    for (0..arr0.dims.len) |ii| {
        if (arr0.dims[ii] != arr1.dims[ii]) {
            return false;
        }
    }

    return true;
}

test "matchArrayDims" {
    var dims0 = [_]usize{ 3, 3, 2 };
    var elems0 = [_]f64{0.0} ** 18;
    var arr0 = try NDArray(f64).init(talloc, elems0[0..], dims0[0..]);
    defer arr0.deinit(talloc);

    var dims1 = [_]usize{ 3, 3, 2 };
    var elems1 = [_]f64{1.0} ** 18;
    var arr1 = try NDArray(f64).init(talloc, elems1[0..], dims1[0..]);
    defer arr1.deinit(talloc);

    var dims2 = [_]usize{ 2, 3, 3 };
    var elems2 = [_]f64{1.0} ** 18;
    var arr2 = try NDArray(f64).init(talloc, elems2[0..], dims2[0..]);
    defer arr2.deinit(talloc);

    var dims3 = [_]usize{ 2, 2, 2 };
    var elems3 = [_]f64{0.0} ** 8;
    var arr3 = try NDArray(f64).init(talloc, elems3[0..], dims3[0..]);
    defer arr3.deinit(talloc);

    try expect(matchArrayDims(f64, &arr0, &arr1));
    try expect(!matchArrayDims(f64, &arr0, &arr2));
    try expect(!matchArrayDims(f64, &arr0, &arr3));
}

const talloc = testing.allocator;

test "getFlatInd" {
    var dims0 = [_]usize{ 2, 3, 3 };
    var elems0 = [_]f64{0.0} ** 18;
    var arr0 = try NDArray(f64).init(talloc, elems0[0..], dims0[0..]);
    defer arr0.deinit(talloc);

    const inds0 = [_]usize{ 1, 2, 1 };
    const exp_ind0: usize = 16; 
	const flat_ind0 = try arr0.getFlatInd(inds0[0..]);
	
	const inds1 = [_]usize{0,0,0};
	const exp_ind1: usize = 0;
	const flat_ind1 = try arr0.getFlatInd(inds1[0..]);

	const inds2 = [_]usize{1,2,2};
	const exp_ind2: usize = 17;
    const flat_ind2 = try arr0.getFlatInd(inds2[0..]);

	try expectEqual(flat_ind0,exp_ind0);
	try expectEqual(flat_ind1,exp_ind1);
	try expectEqual(flat_ind2,exp_ind2);
}

test "calcFlatStride" {
    var dims0 = [_]usize{ 2, 3, 3 };
    var elems0 = [_]f64{0.0} ** 18;
    var arr0 = try NDArray(f64).init(talloc, elems0[0..], dims0[0..]);
    defer arr0.deinit(talloc);
    const check0 = [_]usize{ 9, 3, 1 };

    for (0..3) |aa| {
        try expectEqual(arr0.strides[aa], check0[aa]);
    }

    var dims1 = [_]usize{ 3, 3, 2 };
    var elems1 = [_]f64{0.0} ** 18;
    var arr1 = try NDArray(f64).init(talloc, elems1[0..], dims1[0..]);
    defer arr1.deinit(talloc);
    const check1 = [_]usize{ 6, 2, 1 };

    for (0..3) |aa| {
        try expectEqual(arr1.strides[aa], check1[aa]);
    }
}

test "getEndDimSlice" {
	var dims0 = [_]usize{ 3, 2, 2 };
	var elems0 = [_]f64{0.0} ** 12;
	var arr0 = try NDArray(f64).init(talloc, elems0[0..], dims0[0..]);
	defer arr0.deinit(talloc);

	//print("test: arr0.strides={any}",.{arr0.strides});
	
	var set_inds = [_]usize{1,0,0};
	for (0..dims0[1]) |ii| {
		for (0..dims0[2]) |jj| {
			set_inds[1] = ii;
			set_inds[2] = jj;
			try arr0.set(set_inds[0..],7);
		}
	}

	set_inds[0] = 2;
	for (0..dims0[1]) |ii| {
		for (0..dims0[2]) |jj|{
			set_inds[1] = ii;
			set_inds[2] = jj;
			try arr0.set(set_inds[0..],9);
		}
	}

	const check_arr0 = [_]f64{7} ** 4;
	const check_arr1 = [_]f64{9} ** 4;

	var fixed_inds = [_]usize{1,10,10};
	const ext_slice0 = try arr0.getEndDimSlice(fixed_inds[0..],0);

	fixed_inds[0] = 2;
	const ext_slice1 = try arr0.getEndDimSlice(fixed_inds[0..],0);
 
	// print("test: arr0.elems={any}\n",.{arr0.elems});
	// print("test: check_arr0={any}\n",.{check_arr0});
	// print("test: ext_slice0={any}\n",.{ext_slice0});
	// print("test: check_arr1={any}\n",.{check_arr1});
	// print("test: ext_slice1={any}\n",.{ext_slice1});

    try expectEqualSlices(f64, check_arr0[0..], ext_slice0[0..]);
	try expectEqualSlices(f64, check_arr1[0..], ext_slice1[0..]);
}
