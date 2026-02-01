from setuptools import find_packages, setup

setup(
    name="models",
    version="0.0.1",
    packages=find_packages('./src', exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_dir={
      '': 'src',
    },
)