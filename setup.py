from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_discription = f.read()

setup(
    name='tensorflow_gan_metrics',
    version='0.1.0',
    packages=find_packages(),
    author='sai kumar kella',
    description='Metrics for generative adverserial networks evaluations',
    long_description_content_type = "text/markdown",
    long_description= long_discription,
    license= "MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=[
        "tensorflow",
        "tensorflow_probability"

    ],
)