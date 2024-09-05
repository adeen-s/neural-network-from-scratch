"""Setup script for neural networks from scratch package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nn-scratch",
    version="0.1.0",
    author="Neural Network Enthusiast",
    author_email="nn@example.com",
    description="A neural network implementation from scratch using only NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/nn-scratch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nn-demo=examples.classification_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
