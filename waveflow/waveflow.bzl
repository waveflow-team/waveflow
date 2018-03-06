load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
    "cuda_default_copts",
)

# Waveflow custom rules:
def wf_op_cc_library(name, visibility = [], srcs = [], gpu_srcs = [], deps = []):
  lib_name = "lib%s.so" % name
  cuda_deps = [
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart_static",
  ]

  tf_srcs = [
     "@local_config_tensorflow//:lib/libtensorflow_framework.so"
  ]
  tf_deps = [
    "@local_config_tensorflow//:tensorflow_headers",
    "@local_config_tensorflow//:tensorflow_nsync_headers",
  ]
  tf_copts = [
    "-std=c++11",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-fPIC"
  ]

  if gpu_srcs:
    gpu_target_name = name + "_gpu"
    native.cc_library(
      name = gpu_target_name,
      srcs = gpu_srcs,
      copts = cuda_default_copts,
      deps = deps + if_cuda(cuda_deps)
    )
    cuda_deps = cuda_deps + [gpu_target_name]

  native.cc_binary(
      name = lib_name,
      srcs = srcs + tf_srcs,
      deps = deps + tf_deps + if_cuda(cuda_deps),
      linkshared = 1,
      copts = tf_copts,
  )
  native.alias(
    name = name,
    actual = lib_name
  )


#
#def wf_op_py_library(name, visibility = [], srcs = [], data = [], deps = []):
#  native.py_library(
#      name = name,
#      visibility = visibility,
#      srcs = srcs,
#      data = data,
#      deps = deps
#  )