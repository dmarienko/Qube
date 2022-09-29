import setuptools
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        return "Undefined"


with open('requirements.txt') as f:
    required = f.read().splitlines()


setuptools.setup(
    name="Qube",
    version=get_version("qube/__init__.py"),
    author="Dmitry Marienko",
    author_email="dmitry.ema@gmail.com",
    description="Qube",
    long_description="Quantitative Backtesting Environment",
    long_description_content_type="text/markdown",
    url="https://github.com/dmarienko/Qube",
    project_urls={
        "Bug Tracker": "https://github.com/dmarienko/Qube/issues",
    },
    scripts=['bin/booster'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=['qube/tests']),
    package_data={
        '': ['*.json', '*.csv'],
    },
    python_requires='>=3.6',
    # setup_requires=['setuptools_scm'],
    keywords=["quantitative", "backtesting", "backtester", "quantitative finance"],
    include_package_data=True,
    install_requires=required
)
