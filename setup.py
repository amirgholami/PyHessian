from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='PyHessian',
    url='https://github.com/amirgholami/PyHessian',
    author='Zhewei Yao, Amir Gholami',
    author_email='zheweiy@berkeley.edu, amirgh@berkeley.edu',
    # Needed to actually package something
    packages=['pyhessian'],
    # Needed for dependencies
    install_requires=['numpy', 'torch'],
    # Others
    version='0.0.1',
    license='MIT',
    description='Pytorch library for second-order based analysis and training of Neural Networks',
)