from setup_common import setup_waveflow

setup_waveflow(
    name="waveflow_gpu",
    install_requires=[
        "tensorflow_gpu>=1.6.0"
    ]
)
