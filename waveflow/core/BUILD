"""Waveflow core ops."""

package(default_visibility = [
    "//visibility:private",
])

load(
    "//waveflow:waveflow.bzl",
    "wf_op_cc_library",
)

# public
wf_op_cc_library(
    name = "test",
    srcs = [
        ":op_files",
    ],
    visibility = [
        "//visibility:public",
    ],
)

# private
filegroup(
    name = "op_files",
    srcs = glob([
        "ops/*.cc",
        "kernels/**/*",
    ]),
)
