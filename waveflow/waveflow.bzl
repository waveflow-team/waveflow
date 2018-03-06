load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
    "cuda_default_copts",
)

# Waveflow custom rules:
def wf_op_cc_library(name, visibility = [], srcs = [], deps = []):
  lib_name = "lib%s.so" % name
  cuda_deps = [
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart_static",
  ]
  cuda_copts = ["-DGOOGLE_CUDA=1"]

  tf_srcs = [
     "@local_config_tensorflow//:lib/libtensorflow_framework.so"
  ]
  tf_deps = [
    "@local_config_tensorflow//:tensorflow_headers",
    "@local_config_tensorflow//:tensorflow_nsync_headers",
  ]
  tf_copts = [
    "-std=c++11",
    '-D_GLIBCXX_USE_CXX11_ABI=0',
    '-fPIC'
  ]

  native.cc_binary(
      name = lib_name,
      srcs = srcs + tf_srcs,
      deps = deps + tf_deps + if_cuda(cuda_deps),
      linkshared = 1,
      copts = tf_copts + if_cuda(cuda_copts),
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