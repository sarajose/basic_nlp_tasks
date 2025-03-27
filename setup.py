from setuptools import setup, find_packages

setup(
    name='NLPlib',
    version='0.1.0',
    description='A collection of basic NLP tools and algorithms',
    author='sarajose',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'torch',
        'tensorflow',
        'keras',
        'nltk',
        'conllu',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)