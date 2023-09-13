import setuptools

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="sbrec", 
    version="0.0.1",
    author="ralgond",
    license='MIT',
    author_email="ht201509@gmail.com",
    description="session-based recommenders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ralgond/sbrec",
    packages=setuptools.find_packages(),
    keywords='sasrec, recommendation, sequential',
    classifiers=[
      "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
)