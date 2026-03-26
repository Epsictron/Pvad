from setuptools import setup, find_packages

setup(
    name="speaker_embedding",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
        "onnx": ["onnx>=1.14.0", "onnxruntime>=1.15.0"],
    },
)
