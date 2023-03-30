from setuptools import setup, find_packages

setup(
    name="listener_speaker",
    version="1.0.0",
    description="Listener Speaker Environment",
    author="Di Xue",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gym>=0.12"],
    include_package_data=True,
)
