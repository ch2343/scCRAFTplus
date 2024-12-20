from setuptools import setup, find_packages

setup(
    name='scCRAFTplus',
    version='1.0.0',
    author='Chuan He',
    author_email='ch2343@yale.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scanpy',
        'numpy',
        'tqdm',
        'umap-learn',
        'scipy',
        'pandas',
        'scikit-learn',
        'jax',
        'matplotlib',
        'anndata',
        'scib'
    ],
    url='http://pypi.python.org/pypi/scCRAFTplus/',
    license='LICENSE.txt',
    description='An package for single-cell data integration with additional label using deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
