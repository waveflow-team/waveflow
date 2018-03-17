"""Waveflow python bindings."""

package(default_visibility = [
    "//visibility:private",
])

load(
    "//waveflow:waveflow.bzl",
    "wf_op_py_library",
    "wf_op_py_test",
)

# Targets:
## Build:
wf_op_py_library(
    name = "filters",
    srcs = ["ops/filters.py"],
    data = ["//waveflow/core:libfilters"],
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

wf_op_py_library(
    name = "beamforming",
    srcs = ["ops/beamforming.py"],
    data = ["//waveflow/core:libbeamforming"],
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

## Tests:
wf_op_py_test(
    name = "filters_test",
    srcs = ["ops/filters_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":filters",
        ":test_utils",
        ":utils",
    ],
)

wf_op_py_test(
    name = "beamforming_test",
    srcs = ["ops/beamforming_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":beamforming",
        ":test_utils",
        ":utils",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "op_util.py",
    ],
)

py_library(
    name = "test_utils",
    srcs = [
        "test_util.py",
    ],
)
