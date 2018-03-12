"""Waveflow core ops."""

package(default_visibility = [
    "//visibility:private",
])

load(
    "//waveflow:waveflow.bzl",
    "wf_op_cc_library",
)

# Targets:
## Build:
wf_op_cc_library(
    name = "libfilters",
    srcs = [
        ":filters_src",
    ],
    gpu_srcs = [
        ":filters_gpu_src",
    ],
    visibility = [
        "//visibility:public",
    ],
)

# private
filegroup(
    name = "filters_src",
    srcs = glob(
        [
            "common/**/*",
            "ops/filters_ops.cc",
            "kernels/filters/**/*.cc",
            "kernels/filters/**/*.h",
        ],
        exclude = ["kernels/filters/**/*cu.cc"],
    ),
)

filegroup(
    name = "filters_gpu_src",
    srcs = glob([
        "common/**/*",
        "kernels/filters/**/*cu.cc",
        "kernels/filters/**/*.h",
    ]),
)
