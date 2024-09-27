from setuptools import setup, find_packages

setup(
    name='focus_opt',
    version='0.1.0',
    author='Eliott Kalfon',
    author_email='eliott.kalfon@gmail.com',
    description='Multi-fidelity optimisation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/eliottkalfon/focus-opt',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy'
    ],
)
