from setuptools import find_packages, setup


__version__ = '0.0.3'
URL = 'https://github.com/tzuhanchang/pytorch_hep'

install_requires = [
    'torch',
    'torch_geometric'
]

setup(
    name='torch_hep',
    version=__version__,
    description='Pytorch for high energy physics',
    author='Zihan Zhang',
    author_email='zihan.zhang@cern.ch',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'pytorch',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    # extras_require={
    #     'benchmark': benchmark_requires,
    #     'test': test_requires,
    #     'dev': dev_requires,
    # },
    packages=find_packages(),
    include_package_data=True,
)