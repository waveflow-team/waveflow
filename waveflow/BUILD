package(default_visibility = [
    "//visibility:private",
])

load(
    "//waveflow:waveflow.bzl",
    "wf_toolchain",
)

# Toolchains:
toolchain_type(name = "wf_toolchain_type")

wf_toolchain(
    name = "linux_cpu_impl",
    compiler = "gcc",
)

toolchain(
    name = "linux_cpu_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":linux_cpu_impl",
    toolchain_type = ":wf_toolchain_type",
)

wf_toolchain(
    name = "linux_cuda_impl",
    compiler = "nvcc",
)

toolchain(
    name = "linux_cuda_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":linux_cuda_impl",
    toolchain_type = ":wf_toolchain_type",
)

wf_toolchain(
    name = "windows_cpu_impl",
    compiler = "mvcc",
)

toolchain(
    name = "windows_cpu_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:windows",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:windows",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":windows_cpu_impl",
    toolchain_type = ":wf_toolchain_type",
)

wf_toolchain(
    name = "windows_cuda_impl",
    compiler = "nvcc",
)

toolchain(
    name = "windows_cuda_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:windows",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:windows",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":windows_cuda_impl",
    toolchain_type = ":wf_toolchain_type",
)
