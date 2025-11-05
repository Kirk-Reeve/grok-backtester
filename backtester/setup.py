from setuptools import setup, find_packages

setup(
    name='backtester',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'yfinance==0.2.43',
        'pandas==2.2.3',
        'numpy==2.1.2',
        'matplotlib==3.9.2',
        'joblib==1.4.2',
        'pydantic==2.9.2',
        'pyyaml==6.0.2',
    ],
    entry_points={
        'console_scripts': [
            'backtester = backtester.main:main',
        ],
    },
    author='Your Name',
    description='Advanced Python share trading strategy backtester',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)