from setuptools import setup, find_packages


requirements = [
    'torch'
]

setup(
    name='pytorch_unet',
    version='0.1.0',
    author='Thomas Fan',
    author_email='thomasjpfan@gmail.com',
    url='https://github.come/thomasjpfan/pytorch-unet',
    description='Highly Customizable UNet Implementation',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements
)
