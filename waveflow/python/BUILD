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
### Main:
py_library(
    name = "all",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":envs",
        ":ops",
    ],
)

### Envs:
py_library(
    name = "envs",
    srcs = glob(["envs/*"]),
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

### Ops:
py_library(
    name = "ops",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":beamforming",
        ":filters",
        ":math_ops",
        ":transforms",
    ],
)

wf_op_py_library(
    name = "math_ops",
    srcs = ["ops/math_ops.py"],
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

wf_op_py_library(
    name = "filters",
    srcs = ["ops/filters.py"],
    data = ["//waveflow/core:libfilters"],
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

wf_op_py_library(
    name = "transforms",
    srcs = ["ops/transforms.py"],
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
    name = "math_ops_test",
    srcs = ["ops/math_ops_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "math_ops",
        ":test_utils",
        ":utils",
    ],
)

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
    name = "transforms_test",
    srcs = ["ops/transforms_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":test_utils",
        ":transforms",
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
