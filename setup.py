from setuptools import find_packages, setup


with open("requirements.txt") as requirements:
    requirements = requirements.readlines()
    requirements = [i.strip() for i in requirements]

VERSION = "0.0.1.dev"

setup(
    name="dolomite-engine",
    version=VERSION,
    install_requires=requirements,
    author="Mayank Mishra",
    author_email="mayank.mishra2@ibm.com",
    license="Apache License 2.0",
    url="https://github.com/ibm-granite/dolomite-engine",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
