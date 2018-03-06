package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorflow_headers",
    srcs = [":include_files"],
    hdrs  = [":include_files"],
    includes = ["include"],
    linkstatic = 1,
    visibility = ["//visibility:public"]
)

cc_library(
    name = "tensorflow_nsync_headers",
    includes = ["include/external/nsync/public"],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "libtensorflow_framework.so",
    srcs = ["lib/libtensorflow_framework.so"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

#cc_import(
#    name = "libtensorflow_framework.so",
#    shared_library = "lib/libtensorflow_framework.so",
#    visibility = ["//visibility:public"],
#)

filegroup(
    name = "include_files",
    srcs = glob(["include/**/*"]),
)
