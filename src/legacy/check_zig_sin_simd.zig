const std = @import("std");

const lane_count = 8;
const value_count = 8 * 1024 * 1024;
const repeat_count = 8;
const warmup_value_count = 4096;

const FloatVector = @Vector(lane_count, f32);

const Measurement = struct {
    nanoseconds: i96,
    checksum: f64,
};

fn fillInput(input_values: []f32) void {
    for (input_values, 0..) |*input_value, value_index| {
        const wrapped_index = value_index % 100_000;

        input_value.* =
            @as(f32, @floatFromInt(wrapped_index)) * 0.0001;
    }
}

noinline fn scalarSin(input_value: f32) f32 {
    return @sin(input_value);
}

fn runForcedScalar(
    input_values: []const f32,
    output_values: []f32,
) void {
    std.debug.assert(input_values.len == output_values.len);

    for (0..repeat_count) |repeat_index| {
        const phase =
            @as(f32, @floatFromInt(repeat_index)) * 0.001;

        for (input_values, output_values) |input_value, *output_value| {
            output_value.* += scalarSin(input_value + phase);
        }
    }
}

fn runNormalScalar(
    input_values: []const f32,
    output_values: []f32,
) void {
    std.debug.assert(input_values.len == output_values.len);

    for (0..repeat_count) |repeat_index| {
        const phase =
            @as(f32, @floatFromInt(repeat_index)) * 0.001;

        for (input_values, output_values) |input_value, *output_value| {
            output_value.* += @sin(input_value + phase);
        }
    }
}

fn runSimd(
    input_values: []const f32,
    output_values: []f32,
) void {
    std.debug.assert(input_values.len == output_values.len);
    std.debug.assert(input_values.len % lane_count == 0);

    const vector_count = input_values.len / lane_count;

    for (0..repeat_count) |repeat_index| {
        const phase_scalar =
            @as(f32, @floatFromInt(repeat_index)) * 0.001;

        const phase_vector: FloatVector = @splat(phase_scalar);

        for (0..vector_count) |vector_index| {
            const value_index = vector_index * lane_count;

            const input_pointer: *const [lane_count]f32 =
                @ptrCast(input_values.ptr + value_index);

            const output_pointer: *[lane_count]f32 =
                @ptrCast(output_values.ptr + value_index);

            const input_vector: FloatVector = input_pointer.*;
            const previous_output: FloatVector = output_pointer.*;

            const output_vector =
                previous_output + @sin(input_vector + phase_vector);

            output_pointer.* = output_vector;
        }
    }
}

fn calculateChecksum(values: []const f32) f64 {
    var checksum: f64 = 0.0;

    for (values) |value| {
        checksum += @as(f64, value);
    }

    return checksum;
}

fn measure(
    init: std.process.Init,
    comptime benchmark_function: fn (
        []const f32,
        []f32,
    ) void,
    input_values: []const f32,
    output_values: []f32,
) Measurement {
    @memset(output_values, 0.0);

    const Timestamp = std.Io.Clock.Timestamp;
    const start_timestamp = Timestamp.now(init.io, .awake);

    benchmark_function(
        input_values,
        output_values,
    );

    const end_timestamp = Timestamp.now(init.io, .awake);
    const duration = start_timestamp.durationTo(end_timestamp);

    return .{
        .nanoseconds = duration.raw.nanoseconds,
        .checksum = calculateChecksum(output_values),
    };
}

fn warmup(
    comptime benchmark_function: fn (
        []const f32,
        []f32,
    ) void,
    input_values: []const f32,
    output_values: []f32,
) void {
    @memset(output_values, 0.0);

    benchmark_function(
        input_values,
        output_values,
    );
}

fn nanosecondsToSeconds(nanoseconds: i96) f64 {
    return @as(f64, @floatFromInt(nanoseconds)) / 1.0e9;
}

fn evaluationsPerSecond(
    total_evaluations: f64,
    seconds: f64,
) f64 {
    return total_evaluations / seconds / 1.0e6;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const input_values = try allocator.alloc(
        f32,
        value_count,
    );
    defer allocator.free(input_values);

    const output_values = try allocator.alloc(
        f32,
        value_count,
    );
    defer allocator.free(output_values);

    fillInput(input_values);

    const warmup_input =
        input_values[0..warmup_value_count];

    const warmup_output =
        output_values[0..warmup_value_count];

    warmup(
        runForcedScalar,
        warmup_input,
        warmup_output,
    );

    warmup(
        runNormalScalar,
        warmup_input,
        warmup_output,
    );

    warmup(
        runSimd,
        warmup_input,
        warmup_output,
    );

    const forced_scalar_measurement = measure(
        init,
        runForcedScalar,
        input_values,
        output_values,
    );

    const normal_scalar_measurement = measure(
        init,
        runNormalScalar,
        input_values,
        output_values,
    );

    const simd_measurement = measure(
        init,
        runSimd,
        input_values,
        output_values,
    );

    const total_evaluation_count =
        value_count * repeat_count;

    const total_evaluations: f64 =
        @floatFromInt(total_evaluation_count);

    const forced_scalar_seconds = nanosecondsToSeconds(
        forced_scalar_measurement.nanoseconds,
    );

    const normal_scalar_seconds = nanosecondsToSeconds(
        normal_scalar_measurement.nanoseconds,
    );

    const simd_seconds = nanosecondsToSeconds(
        simd_measurement.nanoseconds,
    );

    const forced_scalar_rate = evaluationsPerSecond(
        total_evaluations,
        forced_scalar_seconds,
    );

    const normal_scalar_rate = evaluationsPerSecond(
        total_evaluations,
        normal_scalar_seconds,
    );

    const simd_rate = evaluationsPerSecond(
        total_evaluations,
        simd_seconds,
    );

    const normal_scalar_speedup_over_forced_scalar =
        forced_scalar_seconds / normal_scalar_seconds;

    const simd_speedup_over_forced_scalar =
        forced_scalar_seconds / simd_seconds;

    const simd_speedup_over_normal_scalar =
        normal_scalar_seconds / simd_seconds;

    const forced_scalar_checksum_difference =
        @abs(
            forced_scalar_measurement.checksum -
                simd_measurement.checksum,
        );

    const normal_scalar_checksum_difference =
        @abs(
            normal_scalar_measurement.checksum -
                simd_measurement.checksum,
        );

    std.debug.print(
        \\Values per repetition: {d}
        \\Repetitions:           {d}
        \\Total evaluations:     {d}
        \\
        \\Forced scalar, noinline:
        \\  Time:                {d:.3} ms
        \\  Throughput:          {d:.3} million sin/s
        \\  Checksum:            {d:.9}
        \\
        \\Normal scalar:
        \\  Time:                {d:.3} ms
        \\  Throughput:          {d:.3} million sin/s
        \\  Checksum:            {d:.9}
        \\
        \\Explicit SIMD ({d}-wide):
        \\  Time:                {d:.3} ms
        \\  Throughput:          {d:.3} million sin/s
        \\  Checksum:            {d:.9}
        \\
        \\Normal scalar / forced scalar speedup: {d:.3}x
        \\SIMD / forced scalar speedup:          {d:.3}x
        \\SIMD / normal scalar speedup:          {d:.3}x
        \\
        \\Forced scalar checksum difference:     {d:.12}
        \\Normal scalar checksum difference:     {d:.12}
        \\
    , .{
        value_count,
        repeat_count,
        total_evaluation_count,

        forced_scalar_seconds * 1000.0,
        forced_scalar_rate,
        forced_scalar_measurement.checksum,

        normal_scalar_seconds * 1000.0,
        normal_scalar_rate,
        normal_scalar_measurement.checksum,

        lane_count,
        simd_seconds * 1000.0,
        simd_rate,
        simd_measurement.checksum,

        normal_scalar_speedup_over_forced_scalar,
        simd_speedup_over_forced_scalar,
        simd_speedup_over_normal_scalar,

        forced_scalar_checksum_difference,
        normal_scalar_checksum_difference,
    });
}
