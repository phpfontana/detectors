from setuptools import setup, find_packages

setup(
    name='detectors',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==2.4.0',
        'torchaudio==2.4.0',
        'torchvision==0.19.0'
    ],
)