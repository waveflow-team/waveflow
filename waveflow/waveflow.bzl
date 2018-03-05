# Toolchain definition:
def _wf_toolchain_impl(ctx):
  toolchain = platform_common.ToolchainInfo(compiler = ctx.attr.compiler)
  return [toolchain]

wf_toolchain = rule(
  _wf_toolchain_impl,
  attrs = {
    'compiler' : attr.string()
  }
)

# Waveflow custom rules:
def wf_op_cc_library(name, visibility = [], srcs = []):
  lib_name = "lib%s.so" % name
  native.cc_binary(
      name = lib_name,
      srcs = srcs + ["@local_config_tensorflow//:lib/libtensorflow_framework.so"],
      deps = [
          "@local_config_tensorflow//:tensorflow_headers",
          "@local_config_tensorflow//:tensorflow_nsync_headers",
      ],
      linkshared = 1,
      copts = [
        "-std=c++11",
        '-D_GLIBCXX_USE_CXX11_ABI=0',
        '-fPIC'
      ],
      linkopts = [
        "-std=c++11",
        '-fPIC'
      ]
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