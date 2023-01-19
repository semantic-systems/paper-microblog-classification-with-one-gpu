import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="microblog_classification_with_one_gpu",
    version="1.0.0",
    author="Junbo Huang",
    author_email="junbo.huang@uni-hamburg.de",
    description="A package for sequence classification",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License Version 2.0",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.7',
)