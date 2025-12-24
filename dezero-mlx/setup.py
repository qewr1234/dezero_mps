from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dezero-mlx",
    version="0.0.13",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning Framework from Scratch with Apple Silicon GPU Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dezero-mlx",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "gpu": ["mlx>=0.0.5"],
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    keywords="deep-learning, machine-learning, neural-network, autograd, mlx, apple-silicon",
)
