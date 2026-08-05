import subprocess
import sys
import shutil
import platform
import importlib.util
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import egg_info
from Cython.Build import cythonize
import numpy

DIST_NAME = "riley-raster"
PROJECT_ROOT = Path(__file__).resolve().parent

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


def lib_link_aliases(
    ext_name: str,
    source_path: Path | None = None,
) -> list[str]:
    lib_names = {lib_link_name(ext_name)}
    if source_path is not None and source_path.suffix == ".zig":
        source_lib = (
            f"{PLATFORM_INFO['lib_prefix']}"
            f"{source_path.stem}"
            f"{PLATFORM_INFO['lib_ext']}"
        )
        lib_names.add(source_lib)
    return sorted(lib_names)

#-------------------------------------------------------------------------------
# Generated package data sync

def ensure_python_package_data() -> None:
    sync_script_path = (
        PROJECT_ROOT / "scripts" / "sync_python_package_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "riley_sync_python_package_data",
        sync_script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load sync script: {sync_script_path}",
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.sync_python_package_data()


class RileyEggInfo(egg_info):

    def run(self):
        ensure_python_package_data()
        super().run()


class RileyBuildPy(build_py):

    def run(self):
        ensure_python_package_data()
        super().run()


#-------------------------------------------------------------------------------
# Custom Multi-Build

class MultiBuildExt(build_ext):

    def run(self):
        ensure_python_package_data()
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
        # Windows/MSVC does not support rpath/runtime_library_dirs
        is_windows = platform.system().lower() == "windows"
        if not is_windows:
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

            if not is_windows:
                if dd not in self.rpath:
                    self.rpath.append(dd)

            for ee in self.extensions:
                if dd not in ee.library_dirs:
                    ee.library_dirs.append(dd)

                if not is_windows:
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
                    zig_src_path = Path(self.get_ext_fullpath(ee.name)).resolve()
                    zig_build_dir = (Path(self.build_lib).resolve()
                                     / self.get_ext_filename(ee.name))
                    zig_build_dir = zig_build_dir.parent
                    for zig_lib_name in lib_link_aliases(
                        ee.name,
                        Path(ee.sources[0]),
                    ):
                        zig_lib_path = zig_build_dir / zig_lib_name
                        zig_src_lib_path = zig_src_path.parent / zig_lib_name
                        shutil.copy2(zig_lib_path,zig_src_lib_path)

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

                            orig_dir = Path(
                                self.get_ext_fullpath(ext_link_lib.name)
                            ).resolve().parent
                            run_dir = Path(
                                self.get_ext_fullpath(ext_with_lib.name)
                            ).resolve().parent
                            for lib_name in lib_link_aliases(
                                ext_link_lib.name,
                                Path(ext_link_lib.sources[0]),
                            ):
                                run_lib_path = run_dir / lib_name
                                orig_lib_path = orig_dir / lib_name

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
            zig_lib_outputs = [
                output_ext_dir / zig_lib_name
                for zig_lib_name in lib_link_aliases(ext.name, first_source_path)
            ]

            print(f"Output zig libraries to:")
            [print(f"    {zig_lib_output}") for zig_lib_output in zig_lib_outputs]
            print(f"Output python extension to:\n    {zig_python_output}")
            print()

            is_windows = platform.system().lower() == "windows"
            zig_target_args = []
            if is_windows:
                # Target MSVC ABI on Windows to ensure CRT compatibility when linking with MSVC
                arch = platform.machine().lower()
                if arch in ("amd64", "x86_64"):
                    target_triple = "x86_64-windows-msvc"
                elif arch in ("arm64", "aarch64"):
                    target_triple = "aarch64-windows-msvc"
                else:
                    target_triple = "i386-windows-msvc"
                zig_target_args = ["-target", target_triple]

            zig_build = [
                "build-lib",
                "-dynamic",
                "-O",
                "ReleaseFast",
                "-lc",
                f"-femit-bin={zig_python_output}",
                *zig_target_args,
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
                for zig_lib_output in zig_lib_outputs:
                    shutil.copy2(zig_python_output,zig_lib_output)
                    print(f"Copied python extension to:\n    {Path(zig_lib_output)}")

                if platform.system().lower() == "windows":
                    # MSVC linker expects "c_riley.lib" when linking against "c_riley"
                    # But Zig creates "[first_source_path.stem].lib" (e.g. "c-riley.lib")
                    # So we copy it to c_riley.lib in the output directory
                    zig_lib_name_win = f"{first_source_path.stem}.lib"
                    zig_lib_path_win = output_ext_dir / zig_lib_name_win
                    target_lib_path_win = output_ext_dir / "c_riley.lib"
                    if zig_lib_path_win.is_file():
                        shutil.copy2(zig_lib_path_win, target_lib_path_win)
                        print(f"Copied import library to:\n    {target_lib_path_win}")

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

H_DIRS = [
    numpy.get_include(),
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "riley" / "cyth"),
    str(PROJECT_ROOT / "src" / "riley" / "zig"),
]

is_windows = platform.system().lower() == "windows"

# Configure compiler flags based on OS to support both MSVC and GCC/Clang
if is_windows:
    cython_compile_args = ["/fp:fast", "/O2"]
    cython_link_args = ["msvcrt.lib", "ucrt.lib", "vcruntime.lib"]
    # Removed -fincremental due to incomplete LLD COFF implementation on Windows
    zig_compile_args = []
    runtime_lib_dirs = []
else:
    cython_compile_args = ["-ffast-math", "-O3"]
    cython_link_args = []
    zig_compile_args = []
    runtime_lib_dirs = [PLATFORM_INFO["runtime_lib_dir"]]

# zig extension
ext_zig = Extension(
    name="riley.zig.c_riley",
    sources=["src/riley/zig/c-riley.zig",],
    extra_compile_args=zig_compile_args,
)

# cython extension linking zig
ext_cython = Extension(
        name="riley.cyth.riley",
        sources=["src/riley/cyth/riley.py",],
        include_dirs=H_DIRS,
        libraries=["c_riley"],
        library_dirs=[],            # populated by run() above
        runtime_library_dirs=runtime_lib_dirs,
        extra_compile_args=cython_compile_args,
        extra_link_args=cython_link_args,
    )

ext_modules = [ext_zig] + cythonize(ext_cython,annotate=True)


#-------------------------------------------------------------------------------
# Setup

setup(
    name=DIST_NAME,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": MultiBuildExt,
        "build_py": RileyBuildPy,
        "egg_info": RileyEggInfo,
    },
    zip_safe=False,
    package_data={
        "riley": [f"*{PLATFORM_INFO['lib_ext']}"],
        "riley.cyth": [f"*{PLATFORM_INFO['lib_ext']}"],
        "riley.zig": [f"*{PLATFORM_INFO['lib_ext']}"],
        "": [f"*{PLATFORM_INFO['lib_ext']}"],
    },
    include_package_data=True,
)
