from setuptools import setup, find_packages

setup(
    name='my-python-project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
    ],
    entry_points={
        'console_scripts': [
            'my-python-project=main:main',  # Adjust this if your main function is located elsewhere
        ],
    },
    author='Kaapeli',
    author_email='your.email@example.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my-python-project',  # Update with your project URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)