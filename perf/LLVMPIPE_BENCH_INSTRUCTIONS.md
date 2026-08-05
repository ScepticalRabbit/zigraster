## LLVMpipe Benchmark Instructions

`temp/bench_llvmpipe.c` now forces software rendering internally at process
startup with:

- `LIBGL_ALWAYS_SOFTWARE=1`
- `GALLIUM_DRIVER=llvmpipe`
- `MESA_LOADER_DRIVER_OVERRIDE=llvmpipe`
- `LP_NUM_THREADS=0`
- `EGL_PLATFORM=surfaceless`

The important detail is `LP_NUM_THREADS=0`. For LLVMpipe this is the true
single-threaded mode and is the correct comparison point for Riley when Riley
is also run single-threaded.

### Expected Renderer

When the benchmark starts it should print:

```text
Renderer: llvmpipe (LLVM ...)
```

If it prints a hardware driver such as `radeonsi`, `iris`, `zink`, or a GPU
name, the run is invalid for LLVMpipe comparison.

### Build And Run

Compile `SSAA=1`:

```bash
cc -O3 -DSSAA_SAMPLES=1 '-DOUT_TAG="ssaa1_llvmpipe"' \
    temp/bench_llvmpipe.c -o temp/bench_llvmpipe -lEGL -lGL -lm
./temp/bench_llvmpipe
```

Compile `SSAA=2`:

```bash
cc -O3 -DSSAA_SAMPLES=4 '-DOUT_TAG="ssaa2_llvmpipe"' \
    temp/bench_llvmpipe.c -o temp/bench_llvmpipe -lEGL -lGL -lm
./temp/bench_llvmpipe
```

`SSAA_SAMPLES=4` corresponds to Riley `SSAA=2`, because each pixel is forced
to evaluate 4 sub-samples before resolve.

### Output Files

The benchmark writes:

- `temp/llvmpipe_stats_median_<tag>.csv`
- `temp/llvmpipe_stats_min_<tag>.csv`
- `temp/llvmpipe_stats_max_<tag>.csv`
- `temp/llvmpipe_stats_mad_<tag>.csv`
- `temp/llvmpipe_stats_cov_<tag>.csv`
- `temp/out_<tag>/`

Examples:

- `temp/llvmpipe_stats_median_ssaa1_llvmpipe.csv`
- `temp/llvmpipe_stats_median_ssaa2_llvmpipe.csv`

### Timing Breakdown

The CSVs include:

- `Clear Time [ms]`
- `Draw Time [ms]`
- `Resolve Time [ms]`
- `E2E Time [ms]`

For the MSAA path, the extra work should mainly appear in `Resolve Time [ms]`
and sometimes in `Draw Time [ms]` depending on the shader.
