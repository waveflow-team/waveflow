load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
    "cuda_default_copts",
)

# Waveflow custom rules:
def wf_op_cc_library(name, visibility = [], srcs = [], gpu_srcs = [], deps = []):
  lib_name = "%s.so" % name
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
  tf_gpu_copts = [ # crosstool_wrapper_driver_is_not_gcc already includes std=c++11 opt
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-fPIC"
  ]

  tf_cc_copts = tf_gpu_copts + [
    "-std=c++11",
  ]

  if gpu_srcs:
    gpu_target_name = name + "_gpu"
    native.cc_library(
      name = gpu_target_name,
      srcs = gpu_srcs,
      copts = tf_gpu_copts + cuda_default_copts(),
      deps = deps + tf_deps +  if_cuda(cuda_deps)
    )
    cuda_deps = cuda_deps + [gpu_target_name]

  native.cc_binary(
      name = lib_name,
      visibility = visibility,
      srcs = srcs + tf_srcs,

      deps = deps + tf_deps + if_cuda(cuda_deps),
      linkshared = 1,
      copts = tf_cc_copts + if_cuda(["-DGOOGLE_CUDA=1"]),
  )
  native.alias(
    visibility = visibility,
    name = name,
    actual = lib_name
  )

def wf_op_py_library(name, visibility = [], srcs = [], data = [], deps = []):
  native.py_library(
      name = name,
      visibility = visibility,
      srcs = srcs,
      data = data,
      deps = deps
  )

def wf_op_py_test(name,
                  visibility = [],
                  srcs = [],
                  main = None,
                  data = [],
                  deps = [],
                  timeout = "short"):
  native.py_test(
    name = name,
      visibility = visibility,
      srcs = srcs,
      main = main,
      data = data,
      deps = deps,
      timeout = timeout
  )