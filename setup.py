from setuptools import find_packages, setup


VERSION = "0.0.1.dev"

setup(
    name="dolomite-engine",
    version=VERSION,
    author="Mayank Mishra",
    author_email="mayank.mishra2@ibm.com",
    url="https://github.com/ibm-granite/dolomite-engine",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
