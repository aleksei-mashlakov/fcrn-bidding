import os

from setuptools import find_packages, setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


install_requires = ["numpy", "pandas", "pytest"]

setup_requirements = ["pytest-runner", "better-setuptools-git-version"]

test_requirements = ["pytest", "nbformat"]

setup(
    author="Aleksei Mashlakov",
    author_email="mashlakov@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
    name="FCRN_bidding",
    # version="0.0.1",
    version_config={"version_format": "{tag}.dev{sha}", "starting_version": "0.0.1"},
    description="Makes bids to the FCR-N market for the battery energy storage ",
    long_description=open("README.md").read(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    install_requires=install_requires,
)
