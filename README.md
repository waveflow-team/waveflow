# Waveflow: signal processing with tensorflow.

Please note, that this project is still on early stage of development and its API may change in future.**

Examples, Tests & Benchmarks
------

You can find examples [here](examples).

To run all waveflow tests:

`bazel test //waveflow/...`

How to build pip package
------

Install required dependencies:
* `bazel >= 0.10.1`: visit https://bazel.build/
* `python >= 3.5` (we suggest you to use conda )
* python modules: `pip install setuptools tensorflow==1.6.0 scipy==1.0.0` 
* `CUDA SDK >= 8.0`

Clone waveflow source code repo:

`git clone git@github.com:waveflow-team/waveflow.git`

Configure project:

```bash
cd waveflow 
./configure
```

Build it:
```bash
# If you want to build with cuda suport: add --config=cuda.
bazel build //waveflow/tools/pip_package:build_pip_package
# You can change the '/tmp' to any destination directory you want.
./bazel-bin/waveflow/tools/pip_package/build_pip_package /tmp
```
The waveflow .whl file should be in  your `/tmp` directory.



