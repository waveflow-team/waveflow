workspace(name = "waveflow")

# Workspace dependencies:
bazel_skylib_version = "34d62c4490826f7642843e0617d7fa614994ef79"
http_archive(
    name = "bazel_skylib",
    strip_prefix = "bazel-skylib-%s" % bazel_skylib_version,
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/%s.tar.gz" % bazel_skylib_version]
)
load("@bazel_skylib//:lib.bzl", "versions")
load("//deps/tensorflow:tensorflow_configure.bzl", "tensorflow_configure")

# Workspace requirements:
versions.check("0.10.1") # min Bazel version

# Toolchains:
register_toolchains(
  '//waveflow:linux_cpu_toolchain',
  '//waveflow:linux_cuda_toolchain',
  '//waveflow:windows_cpu_toolchain',
  '//waveflow:windows_cuda_toolchain'
)

# Waveflow deps:
## Tensorflow
tensorflow_configure(name="local_config_tensorflow")


