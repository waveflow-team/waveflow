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
    deps = [":op_utils"],
)

### Ops:
py_library(
    name = "ops",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":math",
        ":signal",
    ],
)

py_library(
    name = "math",
    srcs = ["ops/math/__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":math_ops",
        ":transform_ops",
        ":unit_ops",
    ],
)

wf_op_py_library(
    name = "math_ops",
    srcs = ["ops/math/math_ops.py"],
    visibility = ["//visibility:public"],
    deps = [":op_utils"],
)

wf_op_py_library(
    name = "transform_ops",
    srcs = ["ops/math/transform_ops.py"],
    visibility = ["//visibility:public"],
    deps = [":op_utils"],
)

wf_op_py_library(
    name = "unit_ops",
    srcs = ["ops/math/unit_ops.py"],
    visibility = ["//visibility:public"],
    deps = [":op_utils"],
)

py_library(
    name = "signal",
    srcs = ["ops/signal/__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":beamforming_ops",
        ":filter_ops",
        ":signal_ops",
    ],
)

wf_op_py_library(
    name = "signal_ops",
    srcs = ["ops/signal/signal_ops.py"],
    visibility = ["//visibility:public"],
    deps = [":op_utils"],
)

wf_op_py_library(
    name = "filter_ops",
    srcs = ["ops/signal/filter_ops.py"],
    data = ["//waveflow/core:libfilters"],
    visibility = ["//visibility:public"],
    deps = [":op_utils"],
)

wf_op_py_library(
    name = "beamforming_ops",
    srcs = ["ops/signal/beamforming_ops.py"],
    data = ["//waveflow/core:libbeamforming"],
    visibility = ["//visibility:public"],
    deps = [":op_utils"],
)

## Tests:
wf_op_py_test(
    name = "math_ops_test",
    srcs = ["ops/math/math_ops_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":math_ops",
        ":test_utils",
        ":op_utils",
    ],
)

wf_op_py_test(
    name = "transform_ops_test",
    srcs = ["ops/math/transform_ops_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":test_utils",
        ":transform_ops",
        ":op_utils",
    ],
)

wf_op_py_test(
    name = "beamforming_ops_test",
    srcs = ["ops/signal/beamforming_ops_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":beamforming_ops",
        ":test_utils",
        ":op_utils",
    ],
)

wf_op_py_test(
    name = "filter_ops_test",
    srcs = ["ops/signal/filter_ops_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":filter_ops",
        ":test_utils",
        ":op_utils",
    ],
)

py_library(
    name = "op_utils",
    srcs = [
        "ops/op_util.py",
    ],
)

py_library(
    name = "test_utils",
    srcs = [
        "test_util.py",
    ],
)
