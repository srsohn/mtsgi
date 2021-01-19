from setuptools import setup, find_packages


setup(
    name="mtsgi",
    version="0.0.1",
    description='Multi-Task Subtask Graph Inference.',
    author='Sungryull Sohn',
    license='MIT',
    packages=['mtsgi'],
    install_requires=[
        'numpy>=1.19',   # TF 2.4 is compatible with 1.19 again
        'matplotlib',
        'gin-config',
        'dm-acme',
        'dm-reverb-nightly==0.2.0.dev20201102',
        'tensorflow==2.4.0',
        'tensorflow_probability>=0.11.0',
        'cloudpickle>=1.3',   # tfp requires 1.3+
        'jax',
        'jaxlib',  # CPU only: we won't be using jax, but ACME depends on jax
        'dm-sonnet>=2.0.0',
        'trfl>=1.1.0',
        'statsmodels',
        'gym==0.13.1',
        'pytest>=5.4.1',
        'pytest-pudb',
        'pytest-mockito',
        'tqdm',
        'graphviz==0.14.2',
        'pybind11==2.6.0',
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)
