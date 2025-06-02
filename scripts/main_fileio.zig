const std = @import("std");
const print = std.debug.print;

const MatAlloc = @import("core/matalloc.zig").MatAlloc;

pub fn main() !void {
    const page_alloc = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(page_alloc);
    defer arena.deinit();

    const arena_alloc = arena.allocator();


    const cwd = std.fs.cwd();

    const dir_name = "file-out";
    const file_name = "data.csv";

    // Make a new directory
    cwd.makeDir(dir_name) catch |err| switch (err) {
        error.PathAlreadyExists => {}, // Path exists do nothing
        else => return err, // Propagate any other error
    };

    // Open the new directory
    var out_dir = try cwd.openDir(dir_name, .{});
    defer out_dir.close();

    // Create a new file
    const csv = try out_dir.createFile(file_name,.{});
    defer csv.close();

    // Create a matrix to write to file
    var mat = try MatAlloc(f64).init(arena_alloc,10,12);
    defer mat.deinit();

    mat.fill(12.7e7);

    // Buffer for writing to file
    var buff: [1024]u8 = undefined;
    @memset(buff[0..], 0);

    for (0..mat.rows_n) |rr| {
        for (0..mat.cols_n) |cc| {
            const str = try std.fmt.bufPrint(&buff, "{},", .{mat.get(rr,cc)});
            _ = try csv.write(str);
        }
        _ = try csv.write("\n");
    }

    print("Finished writing to csv\n",.{});

    // const written = try csv.write("Did this work?");
    // print("Write {d} bytes to file.\n",.{written});


}

// const n = 42;
// var buf: [256]u8 = undefined;
// const str = try std.fmt.bufPrint(&buf, "{}", .{n});
// std.debug.print("{s}\n", .{str});
