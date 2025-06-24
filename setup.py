import subprocess
import sys
import shutil
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy

PACKAGE_NAME = "zigraster"

#-------------------------------------------------------------------------------
# Platform-specific utilities

def get_platform_info() -> dict[str,str]:
    """Get platform-specific file extensions and settings"""
    system = platform.system().lower()

    if system == "windows":
        return {
            "lib_ext": ".dll",
            "lib_prefix": "",
            "runtime_lib_dir": "",  # Windows doesn't use RPATH
        }
    elif system == "darwin":  # macOS
        return {
            "lib_ext": ".dylib",
            "lib_prefix": "lib",
            "runtime_lib_dir": "@loader_path"
        }
    else:  # Linux and other Unix-like
        return {
            "lib_ext": ".so",
            "lib_prefix": "lib",
            "runtime_lib_dir": "$ORIGIN"
        }


PLATFORM_INFO = get_platform_info()

def lib_base_name(ext_full_name: str) -> str:
    return ext_full_name.rsplit(".",maxsplit=1)[-1]

def lib_link_name(ext_name: str) -> str:
    lib_name = lib_base_name(ext_name)
    return f"{PLATFORM_INFO['lib_prefix']}{lib_name}{PLATFORM_INFO['lib_ext']}"

#-------------------------------------------------------------------------------
# Custom Multi-Build

class MultiBuildExt(build_ext):

    def run(self):
        print(80*"=")
        print("MultiBuildExt: run pre-process")
        print(80*"=")

        buil_temp_path = Path(self.build_temp)
        if not buil_temp_path.is_dir():
            buil_temp_path.mkdir(exist_ok=True,parents=True)

        print(f"Creating temp build output directory at:\n    {buil_temp_path}\n")

        build_lib_path = Path(self.build_lib)
        if not build_lib_path.is_dir():
            build_lib_path.mkdir(exist_ok=True,parents=True)

        print(f"Creating library build output directory at:\n    {build_lib_path}\n")

        # Add platform specific runtime paths
        if PLATFORM_INFO["runtime_lib_dir"] not in self.rpath:
            self.rpath.append(PLATFORM_INFO["runtime_lib_dir"])

        # Extract a list of all extension output directories (will be sub
        # directories of the root directories above)
        ext_dirs = []
        for ee in self.extensions:
            ext_path = str(Path(self.get_ext_fullpath(ee.name)).resolve().parent)
            ext_dirs.append(ext_path)

        # Add all extensions specific output directories to all other extensions
        # for libraries and runtime
        for dd in ext_dirs:
            if dd not in self.library_dirs:
                self.library_dirs.append(dd)

            if dd not in self.rpath:
                self.rpath.append(dd)

            for ee in self.extensions:
                if dd not in ee.library_dirs:
                    ee.library_dirs.append(dd)

                if dd not in ee.runtime_library_dirs:
                    ee.runtime_library_dirs.append(dd)

        # Print the configures extensions libraries
        for ee in self.extensions:
            print(80*"-")
            print(f"Directories for extension in 'run': {ee.name}")
            print(2*" "+"include_dirs:")
            [print(6*" " + f"{dd}") for dd in ee.include_dirs]
            print(2*" "+"library_dirs:")
            [print(6*" " + f"{dd}") for dd in ee.library_dirs]
            print(2*" "+"runtime_library_dirs:")
            [print(6*" " + f"{dd}") for dd in ee.runtime_library_dirs]
            print(2*" "+"libraries:")
            [print(6*" " + f"{dd}") for dd in ee.libraries]
            print()

        # Run the standard build process looping over 'build_extension(ext)'
        super().run()

        # Print the global libraries and paths
        print()
        print(80*"-")
        print("Global directories in 'run', post-run")
        print(2*" "+"include_dirs:")
        [print(6*" " + f"{dd}") for dd in self.include_dirs]
        print(2*" "+"library_dirs:")
        [print(6*" " + f"{dd}") for dd in self.library_dirs]
        print(2*" "+"rpath:") # runtime library dirs
        [print(6*" " + f"{dd}") for dd in self.rpath]
        print()

        if self.inplace:
            # Here we need to copy zig libraries to the src directory in-place
            for ee in self.extensions:
                if Path(ee.sources[0]).suffix == ".zig":
                    zig_lib_name = lib_link_name(ee.name)

                    zig_lib_path = (Path(self.build_lib).resolve()
                                    / self.get_ext_filename(ee.name))
                    zig_lib_path = zig_lib_path.parent / zig_lib_name

                    zig_src_path = Path(self.get_ext_fullpath(ee.name)).resolve()
                    zig_src_path = zig_src_path.parent / zig_lib_name
                    shutil.copy2(zig_lib_path,zig_src_path)

        # Make sure linked libraries are in the same folder:
        # 1) loop through all extensions - do they have libraries?
        # 2) if yes, check all other extensions to see if they are the libraries
        # 3) if one extension links to another then copy the built library into
        #    the same directory as the one linking to it
        for ext_with_lib in self.extensions:
            if ext_with_lib.libraries:

                for lib in ext_with_lib.libraries:
                    for ext_link_lib in self.extensions:
                        if (lib == lib_base_name(ext_link_lib.name)
                            or lib == ext_link_lib.name):

                            print("Found extension library cross link:")
                            print(4*" "+ f"{ext_with_lib.name} -> {ext_link_lib.name}")

                            lib_name = lib_link_name(ext_link_lib.name)

                            run_lib_path = Path(self.get_ext_fullpath(ext_with_lib.name))
                            run_lib_path = run_lib_path.resolve().parent / lib_name

                            orig_lib_path = Path(self.get_ext_fullpath(ext_link_lib.name))
                            orig_lib_path = orig_lib_path.resolve().parent / lib_name

                            print("Copying linked extension library:")
                            print(4*" " + f"From: {str(orig_lib_path)}")
                            print(4*" " + f"To  : {str(run_lib_path)}")
                            print()

                            # Need to make sure linked library is in the same
                            # directory as the library looking for it - rpath
                            # is added for linux/mac and windows also looks in
                            # the same directory.
                            shutil.copy2(orig_lib_path,run_lib_path)


    def build_extension(self, ext):
        print(80*"=")
        print("MultiBuildExt: build_extension")
        print(f"Extension = {ext.name}")
        print(80*"=")
        first_source_path = Path(ext.sources[0])

        # Append all extension output directories to all other extensions.
        # This has to be done again as paths change between build and run when
        # the --in-place flag is used!
        ext_dirs = []
        for ee in self.extensions:
            ext_path = str(Path(self.get_ext_fullpath(ee.name))
                                .resolve()
                                .parent)
            ext_dirs.append(ext_path)

        for dd in ext_dirs:
            for ee in self.extensions:
                if dd not in ee.library_dirs:
                    ee.library_dirs.append(dd)

        print(f"Directories for extension in 'build': {ext.name}")
        print(2*" "+"include_dirs:")
        [print(6*" " + f"{dd}") for dd in ext.include_dirs]
        print(2*" "+"library_dirs:")
        [print(6*" " + f"{dd}") for dd in ext.library_dirs]
        print(2*" "+"runtime_library_dirs:")
        [print(6*" " + f"{dd}") for dd in ext.runtime_library_dirs]
        print(2*" "+"libraries:")
        [print(6*" " + f"{dd}") for dd in ext.libraries]
        print()

        output_ext_path = Path(self.get_ext_fullpath(ext.name))
        output_ext_dir = output_ext_path.parent
        if not output_ext_dir.is_dir():
            output_ext_dir.mkdir(exist_ok=True,parents=True)

        print("Creating build output directory at:")
        print(f"    {output_ext_dir}\n")

        if first_source_path.suffix == ".zig":
            assert len(ext.sources) == 1, "Zig compiler expects a single source file"

            print(80*"-")
            print("Zig: Building Extension")
            print(f"{ext.name}")
            print(80*"-")
            print(f"Building with root file:\n    {first_source_path}")

            zig_python_output = self.get_ext_fullpath(ext.name)
            zig_lib_name = lib_link_name(ext.name)
            zig_lib_output = output_ext_dir / zig_lib_name

            print(f"Output zig library to:\n    {zig_lib_output}")
            print(f"Output python extension to:\n    {zig_python_output}")
            print()

            zig_build = [
                "build-lib",
                "-dynamic",
                "-O",
                "ReleaseFast",
                "-lc",
                f"-femit-bin={zig_python_output}",
                *[f"-I{d}" for d in self.include_dirs],
                *ext.extra_compile_args,
                *ext.extra_link_args,
                str(first_source_path),
            ]

            zig_build_str = " ".join(zig_build)

            print(f"Zig build command:\nzig {zig_build_str}\n")


            try:
                # Calls the ziglang pypi package:
                # https://pypi.org/project/ziglang/
                subprocess.check_call([sys.executable, "-m", "ziglang"] + zig_build)
                print("Zig build successful\n")

                # Copy python extension name to linkable library name
                shutil.copy2(zig_python_output,zig_lib_output)
                print(f"Copied python extension to:\n    {Path(zig_lib_output)}")

            except subprocess.CalledProcessError as e:
                print(f"{ext.name}: Zig build failed: {e}")
                raise

        elif (first_source_path.suffix == ".c"
            or first_source_path.suffix == ".pyx"
            or first_source_path.suffix == ".py"):
            print(80*"-")
            print("C/C++/Cython: Build Extension")
            print(f"{ext.name}")
            print(80*"-")

            print(f"{ext.name}: found C/C++/Cython extension using default build process")
            super().build_extension(ext)

        else:
            print(80*"-")
            print("Unrecognised: Default Build Extension")
            print(f"{ext.name}")
            print(80*"-")
            print("Using default build process.")
            super().build_extension(ext)

        print(f"\nbuild_ext complete for: {ext.name}\n")

#-------------------------------------------------------------------------------
# Extensions

H_DIRS = [numpy.get_include(),
          str(Path.cwd()/"src"),
          str(Path.cwd()/"src"/"cyth"),
          str(Path.cwd()/"src"/"zig"),]

# zig extension
ext_zig = Extension(
    name="zigraster.zig.zigraster",
    sources=["src/zigraster/zig/zigraster.zig",],
    extra_compile_args=["-fincremental",],
)

# cython extension linking zig
ext_cython = Extension(
        name="zigraster.cyth.zraster",
        sources=["src/zigraster/cyth/zraster.py",],
        include_dirs=H_DIRS,
        libraries=["zigraster",],  # without the lib and so extension - e.g. libzigarray.so - zig
        library_dirs=[],            # populated by run() above
        runtime_library_dirs=[PLATFORM_INFO["runtime_lib_dir"],],
        extra_compile_args=["-ffast-math",
                            "-O3"],
    )

ext_modules = [ext_zig] + cythonize(ext_cython,annotate=True)


#-------------------------------------------------------------------------------
# Setup

setup(
    name="zigraster",
    ext_modules=ext_modules,
    cmdclass={"build_ext": MultiBuildExt},
    zip_safe=False,
    package_data={
        "zigraster": [f"*{PLATFORM_INFO['lib_ext']}"],
        "zigraster.cython": [f"*{PLATFORM_INFO['lib_ext']}"],
        "zigraster.zig": [f"*{PLATFORM_INFO['lib_ext']}"],
        "": [f"*{PLATFORM_INFO['lib_ext']}"],
    },
    include_package_data=True,
)

