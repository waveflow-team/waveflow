package(default_visibility = [
    "//visibility:private",
])

# Waveflow configuration
config_setting(
    name = "using_cuda",
    define_values = {
        "using_cuda": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_nvcc",
    define_values = {
        "using_cuda": "true",
        "using_cuda_nvcc": "true",
    },
    visibility = ["//visibility:public"],
)

# Targets:
## Build:


py_library(
    name = "py",
    srcs = ["__init__.py"],
    data = ["//waveflow/python:all"],
    deps = [
        "//waveflow/python:all",
    ],
    visibility = ["//visibility:public"],
)
