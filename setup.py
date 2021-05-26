from setuptools import setup, find_packages

setup(
    name='pytorchts',
    version='0.1.0',
    description="PyTorch Probabilistic Time Series Modeling framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",

    url='https://github.com/kashif/pytorch-ts',
    license='MIT',

    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires = [
        'torch==1.4.0',
        'holidays',
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
        'pydantic==1.4.0',
        'matplotlib',
        'python-rapidjson',
        'tensorboard',
    ],

    test_suite='tests',
    tests_require = [
        'flake8',
        'pytest'
    ],
)

#pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
