import setuptools

VERSION = "0.1.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('build.txt', 'wt') as f:
    f.write(VERSION)

setuptools.setup(
    name="CQube",
    version=VERSION,
    author="Dmitry Marienko",
    author_email="dmitry.ema@gmail.com",
    description="Qube",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmarienko/CQube",
    project_urls={
        "Bug Tracker": "https://github.com/dmarienko/CQube/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=required
)
