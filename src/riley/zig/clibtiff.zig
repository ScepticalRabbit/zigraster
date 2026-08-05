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
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const LIBTIFF_PATH = "/usr/lib/x86_64-linux-gnu/libtiff.so.6";

pub const LibTiff = struct {
    dynlib: std.DynLib,

    _TIFFOpen: *const fn ([*:0]const u8, [*:0]const u8) callconv(.c) ?*anyopaque,
    _TIFFClose: *const fn (*anyopaque) callconv(.c) void,
    _TIFFGetField: *const fn (*anyopaque, u32, ...) callconv(.c) c_int,
    _TIFFReadRGBAImage: *const fn (
        *anyopaque,
        u32,
        u32,
        [*]u32,
        c_int,
    ) callconv(.c) c_int,

    pub fn init() !LibTiff {
        var dynlib = std.DynLib.open(LIBTIFF_PATH) catch return error.LibraryNotFound;

        const open_fn = dynlib.lookup(
            *const fn ([*:0]const u8, [*:0]const u8) callconv(.c) ?*anyopaque,
            "TIFFOpen",
        ) orelse return error.SymbolNotFound;
        const close_fn = dynlib.lookup(
            *const fn (*anyopaque) callconv(.c) void,
            "TIFFClose",
        ) orelse return error.SymbolNotFound;
        const get_field_fn = dynlib.lookup(
            *const fn (*anyopaque, u32, ...) callconv(.c) c_int,
            "TIFFGetField",
        ) orelse return error.SymbolNotFound;
        const read_rgba_fn = dynlib.lookup(
            *const fn (*anyopaque, u32, u32, [*]u32, c_int) callconv(.c) c_int,
            "TIFFReadRGBAImage",
        ) orelse return error.SymbolNotFound;

        return LibTiff{
            .dynlib = dynlib,
            ._TIFFOpen = open_fn,
            ._TIFFClose = close_fn,
            ._TIFFGetField = get_field_fn,
            ._TIFFReadRGBAImage = read_rgba_fn,
        };
    }

    pub fn deinit(self: *LibTiff) void {
        self.dynlib.close();
    }

    pub fn open(self: LibTiff, path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque {
        return self._TIFFOpen(path, mode);
    }

    pub fn close(self: LibTiff, tif: *anyopaque) void {
        self._TIFFClose(tif);
    }

    pub fn getField(self: LibTiff, tif: *anyopaque, tag: u32, ptr: anytype) c_int {
        return self._TIFFGetField(tif, tag, ptr);
    }

    pub fn readRGBAImage(
        self: LibTiff,
        tif: *anyopaque,
        w: u32,
        h: u32,
        raster: [*]u32,
        stop: c_int,
    ) c_int {
        return self._TIFFReadRGBAImage(tif, w, h, raster, stop);
    }
};
