// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn sinApproxSIMD(
    comptime N: usize,
    comptime T: type,
    val: @Vector(N, T),
) @Vector(N, T) {
    const reduced_data = reduceTrigInput(N, T, val);
    return sinFromReduced(
        N,
        T,
        reduced_data.reduced,
        reduced_data.quadrant,
    );
}

pub fn cosApproxSIMD(
    comptime N: usize,
    comptime T: type,
    val: @Vector(N, T),
) @Vector(N, T) {
    const reduced_data = reduceTrigInput(N, T, val);
    return cosFromReduced(
        N,
        T,
        reduced_data.reduced,
        reduced_data.quadrant,
    );
}

fn splatValue(
    comptime N: usize,
    comptime T: type,
    value: T,
) @Vector(N, T) {
    return @splat(value);
}

fn trigIntType(comptime T: type) type {
    return switch (T) {
        f32 => i32,
        f64 => i64,
        else => @compileError("maths_simd only supps f32 and f64"),
    };
}

fn TrigReducedType(comptime N: usize, comptime T: type) type {
    return struct {
        reduced: @Vector(N, T),
        quadrant: @Vector(N, trigIntType(T)),
    };
}

fn reduceTrigInput(
    comptime N: usize,
    comptime T: type,
    input: @Vector(N, T),
) TrigReducedType(N, T) {
    const FloatVec = @Vector(N, T);
    const Int = trigIntType(T);
    const IntVec = @Vector(N, Int);

    const inv_half_pi: FloatVec = switch (T) {
        f32 => @splat(0.63661977236758134308),
        f64 => @splat(0.63661977236758134308),
        else => unreachable,
    };

    const quadrant_float: FloatVec = @round(input * inv_half_pi);
    const quadrant_integer: IntVec = @intFromFloat(quadrant_float);

    var reduced = input;
    switch (T) {
        f32 => {
            const half_pi_high: FloatVec =
                @splat(1.57079625129699707031);
            const half_pi_low: FloatVec =
                @splat(7.54978941586159635335e-8);

            reduced = @mulAdd(
                FloatVec,
                -quadrant_float,
                half_pi_high,
                reduced,
            );
            reduced = @mulAdd(
                FloatVec,
                -quadrant_float,
                half_pi_low,
                reduced,
            );
        },
        f64 => {
            const half_pi_high: FloatVec =
                @splat(1.57079632673412561417);
            const half_pi_mid: FloatVec =
                @splat(6.07710050650619224932e-11);
            const half_pi_low: FloatVec =
                @splat(2.02226624879595063154e-21);

            reduced = @mulAdd(
                FloatVec,
                -quadrant_float,
                half_pi_high,
                reduced,
            );
            reduced = @mulAdd(
                FloatVec,
                -quadrant_float,
                half_pi_mid,
                reduced,
            );
            reduced = @mulAdd(
                FloatVec,
                -quadrant_float,
                half_pi_low,
                reduced,
            );
        },
        else => unreachable,
    }

    const quadrant = quadrant_integer & @as(IntVec, @splat(3));
    return .{
        .reduced = reduced,
        .quadrant = quadrant,
    };
}

fn sinReduced(
    comptime N: usize,
    comptime T: type,
    reduced: @Vector(N, T),
) @Vector(N, T) {
    const FloatVec = @Vector(N, T);
    const squared = reduced * reduced;
    var sine_polynomial: FloatVec = undefined;

    switch (T) {
        f32 => {
            sine_polynomial = @splat(1.6059043836821613e-10);
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(-2.505210838544172e-8)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(2.7557319223985893e-6)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(-1.984126984126984e-4)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(8.333333333333333e-3)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(-1.6666666666666666e-1)),
            );
        },
        f64 => {
            sine_polynomial = @splat(-7.6471637318198164759e-13);
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(1.6059043836821614599e-10)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(-2.5052108385441718775e-8)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(2.7557319223985890653e-6)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(-1.9841269841269841253e-4)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(8.3333333333333332177e-3)),
            );
            sine_polynomial = @mulAdd(
                FloatVec,
                squared,
                sine_polynomial,
                @as(FloatVec, @splat(-1.6666666666666665741e-1)),
            );
        },
        else => unreachable,
    }

    return @mulAdd(
        FloatVec,
        reduced * squared,
        sine_polynomial,
        reduced,
    );
}

fn cosReduced(
    comptime N: usize,
    comptime T: type,
    reduced: @Vector(N, T),
) @Vector(N, T) {
    const FloatVec = @Vector(N, T);
    const squared = reduced * reduced;
    var cosine_polynomial: FloatVec = undefined;

    switch (T) {
        f32 => {
            cosine_polynomial = @splat(2.08767569878681e-9);
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-2.755731922398589e-7)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(2.48015873015873e-5)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-1.388888888888889e-3)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(4.1666666666666664e-2)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-5.0e-1)),
            );
        },
        f64 => {
            cosine_polynomial = @splat(4.7794773323873852974e-14);
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-1.1470745597729724714e-11)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(2.0876756987868098979e-9)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-2.7557319223985892511e-7)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(2.4801587301587301587e-5)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-1.3888888888888888889e-3)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(4.1666666666666666667e-2)),
            );
            cosine_polynomial = @mulAdd(
                FloatVec,
                squared,
                cosine_polynomial,
                @as(FloatVec, @splat(-5.0e-1)),
            );
        },
        else => unreachable,
    }

    return @mulAdd(
        FloatVec,
        squared,
        cosine_polynomial,
        splatValue(N, T, 1.0),
    );
}

fn quadrantUseCosine(
    comptime N: usize,
    comptime T: type,
    quadrant: @Vector(N, trigIntType(T)),
) @Vector(N, bool) {
    const IntVec = @Vector(N, trigIntType(T));
    return (quadrant == @as(IntVec, @splat(1))) |
        (quadrant == @as(IntVec, @splat(3)));
}

fn quadrantNegateSin(
    comptime N: usize,
    comptime T: type,
    quadrant: @Vector(N, trigIntType(T)),
) @Vector(N, bool) {
    const IntVec = @Vector(N, trigIntType(T));
    return (quadrant == @as(IntVec, @splat(2))) |
        (quadrant == @as(IntVec, @splat(3)));
}

fn quadrantNegateCos(
    comptime N: usize,
    comptime T: type,
    quadrant: @Vector(N, trigIntType(T)),
) @Vector(N, bool) {
    const IntVec = @Vector(N, trigIntType(T));
    return (quadrant == @as(IntVec, @splat(1))) |
        (quadrant == @as(IntVec, @splat(2)));
}

fn sinFromReduced(
    comptime N: usize,
    comptime T: type,
    reduced: @Vector(N, T),
    quadrant: @Vector(N, trigIntType(T)),
) @Vector(N, T) {
    const use_cosine = quadrantUseCosine(N, T, quadrant);
    const negate_sine = quadrantNegateSin(N, T, quadrant);
    const sine_reduced = sinReduced(N, T, reduced);
    const cosine_reduced = cosReduced(N, T, reduced);
    const sine_magnitude = @select(
        T,
        use_cosine,
        cosine_reduced,
        sine_reduced,
    );
    return @select(
        T,
        negate_sine,
        -sine_magnitude,
        sine_magnitude,
    );
}

fn cosFromReduced(
    comptime N: usize,
    comptime T: type,
    reduced: @Vector(N, T),
    quadrant: @Vector(N, trigIntType(T)),
) @Vector(N, T) {
    const use_cosine = quadrantUseCosine(N, T, quadrant);
    const negate_cosine = quadrantNegateCos(N, T, quadrant);
    const sine_reduced = sinReduced(N, T, reduced);
    const cosine_reduced = cosReduced(N, T, reduced);
    const cosine_magnitude = @select(
        T,
        use_cosine,
        sine_reduced,
        cosine_reduced,
    );
    return @select(
        T,
        negate_cosine,
        -cosine_magnitude,
        cosine_magnitude,
    );
}

fn sinCosApproxPair(
    comptime N: usize,
    comptime T: type,
    input: @Vector(N, T),
) struct {
    sin_val: @Vector(N, T),
    cos_val: @Vector(N, T),
} {
    const reduced_data = reduceTrigInput(N, T, input);
    return .{
        .sin_val = sinFromReduced(
            N,
            T,
            reduced_data.reduced,
            reduced_data.quadrant,
        ),
        .cos_val = cosFromReduced(
            N,
            T,
            reduced_data.reduced,
            reduced_data.quadrant,
        ),
    };
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

const f32_trig_tol: f32 = 1e-5;
const f64_trig_tol: f64 = 1e-12;

test "maths_simd sin and cos approximations f32" {
    const testing = std.testing;
    const N = 4;

    var input_array: [N]f32 = undefined;
    for (0..200) |ii| {
        const float_ii = @as(f32, @floatFromInt(ii));
        const val_0 = -20.0 + float_ii * 0.2;
        const val_1 = val_0 + 0.05;
        const val_2 = val_0 + 0.1;
        const val_3 = val_0 + 0.15;

        input_array[0] = val_0;
        input_array[1] = val_1;
        input_array[2] = val_2;
        input_array[3] = val_3;

        const val_vec: @Vector(N, f32) = input_array;
        const sin_arr: [N]f32 = sinApproxSIMD(N, f32, val_vec);
        const cos_arr: [N]f32 = cosApproxSIMD(N, f32, val_vec);

        for (0..N) |jj| {
            try testing.expectApproxEqAbs(
                @sin(input_array[jj]),
                sin_arr[jj],
                f32_trig_tol,
            );
            try testing.expectApproxEqAbs(
                @cos(input_array[jj]),
                cos_arr[jj],
                f32_trig_tol,
            );
        }
    }
}

test "maths_simd sin and cos approximations f64" {
    const testing = std.testing;
    const N = 4;

    var input_array: [N]f64 = undefined;
    for (0..200) |ii| {
        const float_ii = @as(f64, @floatFromInt(ii));
        const val_0 = -20.0 + float_ii * 0.2;
        const val_1 = val_0 + 0.05;
        const val_2 = val_0 + 0.1;
        const val_3 = val_0 + 0.15;

        input_array[0] = val_0;
        input_array[1] = val_1;
        input_array[2] = val_2;
        input_array[3] = val_3;

        const val_vec: @Vector(N, f64) = input_array;
        const sin_arr: [N]f64 = sinApproxSIMD(N, f64, val_vec);
        const cos_arr: [N]f64 = cosApproxSIMD(N, f64, val_vec);

        for (0..N) |jj| {
            try testing.expectApproxEqAbs(
                @sin(input_array[jj]),
                sin_arr[jj],
                f64_trig_tol,
            );
            try testing.expectApproxEqAbs(
                @cos(input_array[jj]),
                cos_arr[jj],
                f64_trig_tol,
            );
        }
    }
}

test "maths_simd dedicated paths match shared pair f32" {
    const testing = std.testing;
    const N = 4;

    for (0..128) |ii| {
        const float_ii = @as(f32, @floatFromInt(ii));
        const input_vec: @Vector(N, f32) = .{
            -12.0 + float_ii * 0.2,
            -11.95 + float_ii * 0.2,
            -11.9 + float_ii * 0.2,
            -11.85 + float_ii * 0.2,
        };

        const pair = sinCosApproxPair(N, f32, input_vec);
        const sin_arr: [N]f32 = sinApproxSIMD(N, f32, input_vec);
        const cos_arr: [N]f32 = cosApproxSIMD(N, f32, input_vec);
        const pair_sin_arr: [N]f32 = pair.sin_val;
        const pair_cos_arr: [N]f32 = pair.cos_val;

        for (0..N) |jj| {
            try testing.expectApproxEqAbs(
                pair_sin_arr[jj],
                sin_arr[jj],
                f32_trig_tol,
            );
            try testing.expectApproxEqAbs(
                pair_cos_arr[jj],
                cos_arr[jj],
                f32_trig_tol,
            );
        }
    }
}
