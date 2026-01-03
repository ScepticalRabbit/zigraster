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
// dims=(elems_num,3,nodes_per_elem) where 3 is the coord[x,y,z].
pub fn loadFromElemArray(comptime N: usize, 
                         T: type, 
                         elem_array: *const NDArray(T),
                         elem_ind: usize) !Vec3SIMD(N,T) {

    const elem_inds = [_]usize{elem_ind,0,0};
    var start_slice: usize = try elem_array.getFlatInd(elem_inds[0..]);
    var stride: usize = elem_array.strides[1];  

    const x_slice = elem_array.elems[start_slice..start_slice+N];

    start_slice += stride;
    const y_slice = elem_array.elems[start_slice..start_slice+N];

    start_slice += stride;
    const z_slice = elem_array.elems[start_slice..start_slice+N];

    return Vec3SIMD(N,T).init(x_slice,y_slice,z_slice);
}

pub fn saveToElemArray(comptime N: usize,
                       T: type,
                       elem_array: *const NDArray(T),
                       elem_ind: usize,
                       vec: Vec3SIMD(N,T)) !void {
                        
}
