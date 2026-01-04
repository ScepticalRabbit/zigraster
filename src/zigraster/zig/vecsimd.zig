const std = @import("std");
const print = std.debug.print;

pub fn Vec3SIMD(comptime N: usize, comptime T: type) type {
    return struct {
        x: @Vector(N,T),
        y: @Vector(N,T),
        z: @Vector(N,T),

        const Self = @This();

        pub fn init(x: []const T, y: []const T, z: []const T) Self {
            return .{
                .x = x[0..N].*,
                .y = y[0..N].*,
                .z = z[0..N].*,
            };
        }

        pub fn initSplat(x_splat: T, y_splat: T, z_splat: T) Self {
            return .{
                .x = @splat(x_splat),
                .y = @splat(y_splat),
                .z = @splat(z_splat),    
            };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{
              .x = self.x + other.x,
              .y = self.y + other.y,
              .z = self.z + other.z,  
            };
        }
        
        pub fn sub(self: Self, other: Self) Self {
            return .{
              .x = self.x - other.x,
              .y = self.y - other.y,
              .z = self.z - other.z,  
            };
        }

        pub fn scale(self: Self, scale: T) Self {
            const scale_vec: @Vector(N, T) = @splat(scale);
            return .{
              .x = self.x * scale_vec,
              .y = self.y * scale_vec,
              .z = self.z * scale_vec,  
            };
        }

        pub fn dot(self: Self, other: Self) @Vector(N,T) {
            return (self.x * other.x) + (self.y * other.y) + (self.z * other.z);
        }

        pub fn vecLenSquare(self: Self) @Vector(N,T) {
            return self.dot(self);
        }

        pub fn vecLen(self: Self) @Vector(N,T) {
            return @sqrt(self.dot(self));
        }

        pub fn cross(self: Self, other: Self) Self {
            return .{
                .x = (self.y * other.z) - (self.z * other.y),
                .y = (self.z * other.x) - (self.x * other.z),
                .z = (self.x * other.y) - (self.y * other.x),
            };
        }
    };
}



// NOTE: this is not a general function, it only works with NDArrays representing a mesh with
// dims=(elems_num,3,nodes_per_elem) where 3 is the coord[x,y,z]. N should be nodes_per_elem.
pub fn loadVec3FromElemArray(comptime N: usize, 
                             comptime T: type, 
                             elem_array: *const NDArray(T),
                             elem_ind: usize) !Vec3SIMD(N,T) {

    //const elem_inds = [_]usize{elem_ind,0,0};
    var start_slice: usize = try elem_array.getFlatInd(&[_]usize{elem_ind,0,0});
    // if coords then stride=3, if fields then stride=fields_num
    const stride: usize = elem_array.strides[1];  

    const x_slice = elem_array.elems[start_slice..start_slice+N];

    start_slice += stride;
    const y_slice = elem_array.elems[start_slice..start_slice+N];

    start_slice += stride;
    const z_slice = elem_array.elems[start_slice..start_slice+N];

    return Vec3SIMD(N,T).init(x_slice,y_slice,z_slice);
}

// NOTE: this is not a general function, it only works with NDArrays representing a mesh with
// dims=(elems_num,3,nodes_per_elem) where 3 is the coord[x,y,z]. N should be nodes_per_elem.
pub fn saveVec3ToElemArray(comptime N: usize,
                           comptime T: type,
                           elem_array: *const NDArray(T),
                           elem_ind: usize,
                           vec: Vec3SIMD(N,T)) !void {
                       
    const elem_inds = [_]usize{elem_ind,0,0};
    var start_slice: usize = try elem_array.getFlatInd(elem_inds[0..]);
    // if coords then stride=3, if fields then stride=fields_num
    const stride: usize = elem_array.strides[1];

    elem_array.elems[start_slice..start_slice+N].* = vec.x;
    start_slice += stride;
    elem_array.elems[start_slice..start_slice+N].* = vec.y;
    start_slice += stride;
    elem_array.elems[start_slice..start_slice+N].* = vec.z;        
}

pub fn mat44Mul(comptime N: usize,
                T: type,
                mat: Mat44T(ElemType), 
                vec: Vec3SIMD(ElemType)
                ) Vec3SIMD(N,T) {
        
    var vec_res: Vec3SIMD(N,D) = undefined;  

    mat_row = Vec3SIMD.initSplat(mat.get(0,0),mat.get(0,1),mat.get(0,2));
    vec_res.x = mat_row.dot(vec) + @splat(mat.get(row,3));

    mat_row = Vec3SIMD.initSplat(mat.get(1,0),mat.get(1,1),mat.get(1,2));
    vec_res.y = mat_row.dot(vec) + @splat(mat.get(row,3));

    mat_row = Vec3SIMD.initSplat(mat.get(2,0),mat.get(2,1),mat.get(2,2));
    vec_res.z = mat_row.dot(vec) + @splat(mat.get(row,3));

    return vec_res;
}

pub fn VecSIMD(comptime D: usize, comptime N: usize, comptime T: type) type {
    return struct {
        elems: [D]@Vector(N,T),

        const Self = @This();

        pub fn init(slices: [D][]const T) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.elems[ii] = slices[ii][0..N].*;
            }
            return out;
        }

        pub fn initSplat(splat_vals: [D]T) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.elems[ii] = @splat(splat_vals[ii]);
            }
            return out;
        }

        pub fn add(self: Self, other: Self) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.elems[ii] = self.elems[ii] + other.elems[ii];                
            }
            return out;
        }
        
        pub fn sub(self: Self, other: Self) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.elems[ii] = self.elems[ii] - other.elems[ii];                
            }
            return out;
        }

        pub fn scale(self: Self, scale: T) Self {
            var out: Self = undefined;
            const scale_vec: @Vector(N,T) = @splate(scale);
            inline for (0..D) |ii| {
                out.elems[ii] = scale_vec * self.elems[ii];                
            }
            return out;
        }

        pub fn dot(self: Self, other: Self) @Vector(N,T) {
            var out: @Vector(N,T) = @splat(0);
            inline for (0..D) |ii| {
                out += self.elems[ii] * other.elems[ii];                
            }
            return out;
        }

        pub fn vecLenSquare(self: Self) @Vector(N,T) {
            return self.dot(self);
        }

        pub fn vecLen(self: Self) @Vector(N,T) {
            return @sqrt(self.dot(self));
        }
    };
}

pub fn loadVecFromElemArray(comptime D: usize,
                            comptime N: usize, 
                            comptime T: type, 
                            elem_array: *const NDArray(T),
                            elem_ind: usize) !Vec3SIMD(D,N,T) {
    const stride = elem_array.strides[1];
    const flat_start = try elem_array.getFlatInd(&[_]usize{elem_ind,0,0});

    var out: VecSIMD(D,N,T) = undefined;
    inline for (0..D) |ii| {
        const start_slice = flat_start + (ii*stride);
        out.elems[ii] = elem_array.elems[start_slice..start_slice+N].*;
    }
    return res;
}

pub fn saveVecToElemArray(comptime D: usize, 
                           comptime N: usize,
                           comptime T: type,
                           elem_array: *const NDArray(T),
                           elem_ind: usize,
                           vec: VecSIMD(D,N,T)) !void {
    const stride = elem_array.strides[1];
    const flat_start = try elem_array.getFlatInd(&[_]usize{elem_ind,0,0})                      

    inline for (0..D) |ii| {
        const start_slice = flat_start + (ii*stride);
        elem_array.elems[start_slice..start_slice+N].* = vec.elems[ii];
    }
    return res;
}
