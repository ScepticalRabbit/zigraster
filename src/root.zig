const std = @import("std");
const testing = std.testing;

pub export fn addInts(a: i32, b: i32) i32 {
    return a + b;
}

pub export fn subInts(a: i32, b: i32) i32 {
    return a - b;
}

pub export fn giveBack(a: i32) i32 {
    return a;
}

test "basic add functionality" {
    try testing.expect(addInts(3, 7) == 10);
}
