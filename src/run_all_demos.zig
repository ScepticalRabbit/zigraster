const std = @import("std");

const demo_sphere200 = @import("demo_sphere200.zig");
const demo_psf = @import("demo_psf.zig");
const demo_rabbits = @import("demo_rabbits.zig");
const demo_rabbits_fields = @import("demo_rabbits_fields.zig");
const demo_rabbits_rgb = @import("demo_rabbits_rgb.zig");
const demo_dicuq = @import("demo_dicuq.zig");
const demo_stereocal = @import("demo_stereocal.zig");

pub fn main(init: std.process.Init) !void {
    try demo_sphere200.main(init);
    try demo_psf.main(init);
    try demo_rabbits.main(init);
    try demo_rabbits_rgb.main(init);
    try demo_rabbits_fields.main(init);
    try demo_dicuq.main(init);
    try demo_stereocal.main(init);
}
