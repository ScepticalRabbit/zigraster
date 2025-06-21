const std = @import("std");
const testing = std.testing;

pub fn CMatrix(comptime ElemType: type) type {
    return extern struct {
        elems: [*c]ElemType,
        dims: [*c]usize,
        numel: usize,
        ndim: usize,
    };
}

const CMatrixF64 = CMatrix(f64);
const CMatrixUS = CMatrix(usize);

pub fn ZMatrix(comptime ElemType: type) type {
    return extern struct {
        elems: []ElemType,
        dims: []usize,
        numel: usize,
        ndim: usize,
    };
}