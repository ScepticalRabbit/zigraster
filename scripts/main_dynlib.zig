const std = @import("std");
const print = std.debug.print;

pub fn main() !void {
    const dyn_lib_path = "zig-out/lib/libzigraster.so";
    print("Dynamic lib path: {s}\n",.{dyn_lib_path});

    var dyn_lib = try std.DynLib.open(dyn_lib_path);
    defer dyn_lib.close();

    const give = dyn_lib.lookup(
        *const fn (i32) callconv(.C) i32,
        "giveBack",
    ) orelse return error.NoFunction;

    const addInts = dyn_lib.lookup(
        *const fn (i32,i32) callconv(.C) i32,
        "addInts",
    ) orelse return error.NoFunction;

    const subInts = dyn_lib.lookup(
        *const fn (i32,i32) callconv(.C) i32,
        "subInts",
    ) orelse return error.NoFunction;

    const a: i32 = 1;
    const b: i32 = 2;
    print("Function give: {}\n",.{give});
    print("Dynamic lib give: {d}\n",.{give(a)});
    print("Dynamic lib add: 1+2={d}\n",.{addInts(a,b)});
    print("Dynamic lib sub: 1-2={d}\n\n",.{subInts(a,b)});
}