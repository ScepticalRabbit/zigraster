const std = @import("std");

const demo0_quickstart = @import("demo0_quickstart.zig");
const demo1_sphere200 = @import("demo1_sphere200.zig");
const demo2_psf = @import("demo2_psf.zig");
const demo3_rabbits = @import("demo3_rabbits.zig");
const demo4_rabbits_rgb = @import("demo4_rabbits_rgb.zig");
const demo5_rabbits_fields = @import("demo5_rabbits_fields.zig");
const demo6_dicuq = @import("demo6_dicuq.zig");
const demo8_stereocal = @import("demo8_stereocal.zig");
const demo9_feature_zoo = @import("demo9_feature_zoo.zig");

pub fn main(init: std.process.Init) !void {
    try demo0_quickstart.main(init);
    try demo1_sphere200.main(init);
    try demo2_psf.main(init);
    try demo3_rabbits.main(init);
    try demo4_rabbits_rgb.main(init);
    try demo5_rabbits_fields.main(init);
    try demo6_dicuq.main(init);
    try demo8_stereocal.main(init);
    try demo9_feature_zoo.main(init);
}
