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
)

## Tests:
wf_op_py_test(
    name = "filters_test",
    srcs = ["ops/filters_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":filters",
        ":test_utils"
    ],
)

py_library(
    name = "test_utils",
    srcs = glob([
        "test_util.py",
        "op_util.py"
    ]),
)
