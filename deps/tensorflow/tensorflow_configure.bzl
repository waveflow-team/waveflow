_TF_INCLUDE_PATH = "TF_INCLUDE_PATH"
_TF_LIB_PATH = "TF_LIB_PATH"

def _get_env_var_with_default(repository_ctx, env_var):
  """Returns evironment variable value."""
  if env_var in repository_ctx.os.environ:
    value = repository_ctx.os.environ[env_var]
    return value
  else:
    fail("Environment variable '%s' was not set." % env_var)

def _get_tf_conf(repository_ctx):
  """Returns structure containing all required information about tensorflow
     configuration on host platform.
  """
  include_path = _get_env_var_with_default(repository_ctx, _TF_INCLUDE_PATH)
  lib_path = _get_env_var_with_default(repository_ctx, _TF_LIB_PATH)
  return struct(
    include_path = include_path,
    lib_path = lib_path
  )

def _tensorflow_autoconf_impl(repository_ctx):
  """Implementation of tensorflow autoconf. rule."""
  tf_conf = _get_tf_conf(repository_ctx)
  print("Using %s=%s" % (_TF_INCLUDE_PATH, tf_conf.include_path))
  print("Using %s=%s" % (_TF_LIB_PATH, tf_conf.lib_path))
  repository_ctx.symlink(tf_conf.include_path, 'include')
  repository_ctx.symlink(tf_conf.lib_path, 'lib')
  repository_ctx.template('BUILD', Label("//deps/tensorflow:tensorflow.BUILD"))


tensorflow_configure = repository_rule(
  implementation = _tensorflow_autoconf_impl,
  environ = [
    _TF_INCLUDE_PATH,
    _TF_LIB_PATH
  ]
)