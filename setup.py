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
    name="CQube",
    version=get_version("qube/__init__.py"),
    author="Dmitry Marienko",
    author_email="dmitry.ema@gmail.com",
    description="Qube",
    long_description="Quantitative Backtesting Environment",
    long_description_content_type="text/markdown",
    url="https://github.com/dmarienko/CQube",
    project_urls={
        "Bug Tracker": "https://github.com/dmarienko/CQube/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=['tests']),
    package_data={
        'configs': ['config-default/*.*'],
        'portfolio': ['report_templates/*.*']
    },
    python_requires='>=3.6',
    # setup_requires=['setuptools_scm'],
    keywords=["quantitative", "backtesting", "backtester", "quantitative finance"],
    include_package_data=True,
    install_requires=required
)
