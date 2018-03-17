package(default_visibility = [
    "//visibility:private",
])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "setup.py",
        "LICENSE.txt",
        "README.rst",
        "//waveflow:py",
    ],
    visibility = ["//visibility:public"],
)
