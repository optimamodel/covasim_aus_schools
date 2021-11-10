from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="covasim_aus_schools",
    version="1.0.0",
    url="",
    install_requires=requirements,
)
