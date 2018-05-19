package(default_visibility = [
    "//visibility:private",
])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    args = select({
        "//waveflow:using_cuda": ["setup_gpu.py"],
        "//waveflow:using_cuda_nvcc": ["setup_gpu.py"],
        "//conditions:default": ["setup_cpu.py"],
    }),
    data = [
        "LICENSE.txt",
        "MANIFEST.in",
        "README.rst",
        "//waveflow:py",
        "setup_common.py",
    ] + select({
        "//waveflow:using_cuda": ["setup_gpu.py"],
        "//waveflow:using_cuda_nvcc": ["setup_gpu.py"],
        "//conditions:default": ["setup_cpu.py"],
    }),
    visibility = ["//visibility:public"],
)
