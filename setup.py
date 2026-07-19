from setuptools import setup, find_packages

setup(
    name='dsna',
    version='0.1.0',
    description='Dual-System Neural Architecture: Combining Modular Skills with a Shared Global Workspace',
    author='DSNA Team',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'gym>=0.21.0',
        'gym-minigrid>=1.2.0',
        'babyai>=1.1.0',
        'pyyaml>=5.4.0',
        'tensorboard>=2.5.0',
    ],
    python_requires='>=3.7',
)
