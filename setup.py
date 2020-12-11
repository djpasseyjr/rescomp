import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rescomp",
    version="0.0.1",
    author="DJ Passey",
    author_email="djpasseyjr@unc.edu",
    description="A reservoir computer and chaotic systems package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djpasseyjr/rescomp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Machine Learning :: Time Series"
    ],
    install_requires=[
        'numpy',
        'networkx',
        'scipy',
        'warnings'
        'math',
        'numdifftools',
        'matplotlib',
        'findiff'
    ],
    python_requires='>=3.6',
    test_suite='nose.collector',
    tests_require=['nose', 'itertools'],
    include_package_data=True
)
