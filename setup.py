
# ext_cython = Extension(
#         "pyvale.cython.rastercyth",
#         ["src/pyvale/cython/rastercyth.py",],
#         include_dirs=[numpy.get_include()],
#         extra_compile_args=["-ffast-math",openmp_arg],
#         extra_link_args=[openmp_arg],
#     )

# ext_dic = Extension(
#     'pyvale.dic.dic2dcpp',
#     sorted(glob("src/pyvale/dic/cpp/dic*.cpp")),
#     language="c++",
#     include_dirs=[pybind11.get_include()],
#     extra_compile_args=['-g', '-O0', '-fopenmp'] if debug_mode else ['-O3', '-fopenmp'],
#     extra_link_args=['-fopenmp'] + (['-g'] if debug_mode else []),
# )
# ext = cythonize([ext_cython], annotate=True) + [ext_dic]