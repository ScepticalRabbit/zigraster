const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const libcheck = b.addSharedLibrary(.{
        .name = "zigraster",
        .root_source_file = b.path("src/zigraster/zig/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(libcheck);
}
