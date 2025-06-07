const std = @import("std");
const print = std.debug.print;

const MatAlloc = @import("core/matalloc.zig").MatAlloc;

pub fn main() !void {
    const page_alloc = std.heap.page_allocator;

    var arena = std.heap.ArenaAllocator.init(page_alloc);
    defer arena.deinit();

    const arena_alloc = arena.allocator();

    const mat0 = try MatAlloc(f64).init(arena_alloc,10,7);
    const mat1 = try MatAlloc(f64).init(arena_alloc,6,10);

    mat0.fill(0.0);
    mat1.fill(1.0);

    var frames = std.ArrayList(MatAlloc(f64)).init(arena_alloc);

    try frames.append(mat0);
    try frames.append(mat1);

    print("\nMat0:\n",.{});
    frames.items[0].matPrint();

    print("\nMat1:\n",.{});
    frames.items[1].matPrint();

    print("Success!\n",.{});
}