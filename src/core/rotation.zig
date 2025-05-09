const std = @import("std");
const print = std.debug.print;
const expectEqual = std.testing.expectEqual;
const Mat33f = @import("matrix.zig").Mat33f;


pub const Rotation = struct {
    alpha_z: f64 = 0.0,
    beta_y: f64 = 0.0,
    gamma_x: f64 = 0.0,
    matrix: Mat33f = undefined,

    const Self = @This();

    // const rows_n: usize = 3;
    // const cols_n: usize = 3;

    pub fn init(alpha_z: f64, beta_y: f64, gamma_x: f64) Rotation {
        var rot = Rotation{ .alpha_z = alpha_z, .beta_y = beta_y, .gamma_x = gamma_x };
        rot.calcRotMat();
        return rot;
    }

    pub fn calcRotMat(self: *Rotation) void {
        // NOTE: this is equivalent to ZYX for intrinsic (one after the other) not zyx which is extrinsic (all relative to global)

        // Row major as in C
        // Row 1
        self.matrix.elems[0] = @cos(self.alpha_z) * @cos(self.beta_y);
        self.matrix.elems[1] = @cos(self.alpha_z) * @sin(self.beta_y) * @sin(self.gamma_x) - @sin(self.alpha_z) * @cos(self.gamma_x);
        self.matrix.elems[2] = @cos(self.alpha_z) * @sin(self.beta_y) * @cos(self.gamma_x) + @sin(self.alpha_z) * @sin(self.gamma_x);
        // Row 2
        self.matrix.elems[3] = @sin(self.alpha_z) * @cos(self.beta_y);
        self.matrix.elems[4] = @sin(self.alpha_z) * @sin(self.beta_y) * @sin(self.gamma_x) + @cos(self.alpha_z) * @cos(self.gamma_x);
        self.matrix.elems[5] = @sin(self.alpha_z) * @sin(self.beta_y) * @cos(self.gamma_x) - @cos(self.alpha_z) * @sin(self.gamma_x);
        // Row 3
        self.matrix.elems[6] = -@sin(self.beta_y);
        self.matrix.elems[7] = @cos(self.beta_y) * @sin(self.gamma_x);
        self.matrix.elems[8] = @cos(self.beta_y) * @cos(self.gamma_x);
    }

    pub fn matPrint(self: *const Rotation) void {
        self.matrix.matPrint();
    }
};

// NOTE: Check this
// Intrinsic (z-y-x) Rotation Matrix: applied one by one
// [ cos(beta)cos(gamma)             cos(beta)sin(gamma)             sin(beta)          ]
// [ sin(alpha)sin(beta)cos(gamma) - cos(alpha)sin(gamma)   sin(alpha)sin(beta)sin(gamma) + cos(alpha)cos(gamma)   -sin(alpha)cos(beta) ]
// [ -cos(alpha)sin(beta)cos(gamma) - sin(alpha)sin(gamma)  -cos(alpha)sin(beta)sin(gamma) + sin(alpha)cos(gamma)    cos(alpha)cos(beta)  ]

// Extrinsic (z-y-x) Rotation Matrix: relative to global axis
// [ cos(gamma)cos(beta)             cos(gamma)sin(beta)sin(alpha) - sin(gamma)cos(alpha)   cos(gamma)sin(beta)cos(alpha) + sin(gamma)sin(alpha) ]
// [ sin(gamma)cos(beta)             sin(gamma)sin(beta)sin(alpha) + cos(gamma)cos(alpha)   sin(gamma)sin(beta)cos(alpha) - cos(gamma)sin(alpha) ]
// [ -sin(beta)                    cos(beta)sin(alpha)                              cos(beta)cos(alpha)                              ]
