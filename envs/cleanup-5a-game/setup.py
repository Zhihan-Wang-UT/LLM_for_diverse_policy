from setuptools import setup, find_packages


setup(
    name="cleanup5agame",
    version="0.0.1",
    description="Clean up game environment with on 5 actions",
    url="https://github.com/semitable/matrix-games",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gym>=0.12", "namespace"],
    include_package_data=True,
)


