from setuptools import setup, Extension

fuzzy_module = Extension("fuzzy", sources=["fuzzy.c"])

setup(
    name="remove_repeating_suffix",
    version="1.0",
    description="Python package with C extensions to remove repeating suffixes",
    ext_modules=[fuzzy_module],
)
