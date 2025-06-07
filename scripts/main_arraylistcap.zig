const std = @import("std");
const print = std.debug.print;

pub fn main() !void {
    const page_alloc = std.heap.page_allocator;

    var arena = std.heap.ArenaAllocator.init(page_alloc);
    defer arena.deinit();

    const arena_alloc = arena.allocator();


    var buffer = try std.ArrayList(u8).initCapacity(arena_alloc, 7);
    defer buffer.deinit();

    for (0..100) |ii| {
        _ = ii;
        try buffer.append('a');
    }
}