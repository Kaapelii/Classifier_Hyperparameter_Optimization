from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Classifier_Hyperparameter_Optimization',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'my-python-project=main:main',  # Adjust this if your main function is located elsewhere
        ],
    },
    author='Kaapeli',
    author_email='',
    description='Train and optimize SVM classifier pipelines using various hyperparameter optimization techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Kaapelii/Classifier_Hyperparameter_Optimization',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)