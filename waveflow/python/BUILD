"""Waveflow python bindings."""

package(default_visibility = [
    "//visibility:private",
])

#load(
#    "//waveflow:waveflow.bzl",
#    "wf_op_py_library",
#)

# Rules:
## public:
#wf_op_py_library(
#    name = "lib",
#    srcs = [":all_files"],
#    data = ["//waveflow/core:test"],
#    visibility = ["//visibility:public"],
#    #   deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
#)

## private:
filegroup(
    name = "all_files",
    srcs = glob([
        "waveflow/**/*",
    ]),
)
