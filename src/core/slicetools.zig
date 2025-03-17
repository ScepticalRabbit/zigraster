
pub fn max(comptime EType: type, slice: []const EType) EType {
    var max_out: EType = slice[0];
    for (1..slice.len) |ii| {
        if (slice[ii] > max_out) {
            max_out = slice[ii];
        }
    }
    return max_out;
}

pub fn min(comptime EType: type, slice: []const EType) EType {
    var min_out: EType = slice[0];
    for (1..slice.len) |ii| {
        if (slice[ii] < min_out) {
            min_out = slice[ii];
        }
    }
    return min_out;
}

pub fn sum(comptime EType: type, slice: []const EType) EType {
    var sum_out: EType = 0;
    for (0..slice.len) |ii| {
        sum_out += slice[ii];
    }
    return sum_out;
}

pub fn mean(comptime EType: type, slice: []const EType) EType {
    return sum(EType, slice) / slice.len;
}