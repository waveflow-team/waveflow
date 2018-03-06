workspace(name = "waveflow")

# Workspace dependencies:
bazel_skylib_version = "34d62c4490826f7642843e0617d7fa614994ef79"
http_archive(
    name = "bazel_skylib",
    strip_prefix = "bazel-skylib-%s" % bazel_skylib_version,
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/%s.tar.gz" % bazel_skylib_version]
)
load("@bazel_skylib//:lib.bzl", "versions")
load("//third_party/tensorflow:tensorflow_configure.bzl", "tensorflow_configure")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")

# Workspace requirements:
versions.check("0.10.1") # min Bazel version

# Waveflow deps:
cuda_configure(name="local_config_cuda")
tensorflow_configure(name="local_config_tensorflow")



