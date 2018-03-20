Waveflow: signal processing with tensorflow.

To run all tests:
bazel test //waveflow/...

To build pip package run:
bazel build //waveflow/tools/pip_package:build_pip_package &&
./bazel-bin/waveflow/tools/pip_package/build_pip_package /path/to/destionation/dir
