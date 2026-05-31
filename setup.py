from setuptools import setup, find_packages

setup(
    name='qubo',
    version='0.1.0',
    description='A developer-friendly quantum simulator toolkit',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'click',
        'pytest',
        'jupyter',
        'PyQt5'
    ],
    entry_points={
        'console_scripts': [
            'qubo=qubo.cli:main',
        ],
    },
)
