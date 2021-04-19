from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
   name='DRhard',
   version='0.1.0',
   description='Optimizing Dense Retrieval Model Training with Hard Negatives',
   url='https://github.com/jingtaozhan/DRhard',
   classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD-3-Clause License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
   license="BSD-3-Clause License",
   long_description=readme,
   install_requires=[
        'torch==1.7.0', 
        'transformers==3.4.0', 
        'faiss-gpu==1.6.4.post2',
        'tensorboard==2.3.0'
    ],
)