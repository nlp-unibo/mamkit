from pathlib import Path

from setuptools import setup

readme_path = Path(__file__).absolute().parent.joinpath('README.md')

with readme_path.open('r', encoding='utf-8') as fh:
    long_description = fh.read()

requirements_path = Path(__file__).absolute().parent.joinpath('requirements.txt')

with requirements_path.open('r') as f:
    requirements = f.readlines()
    requirements = [req for req in requirements if "--hash" not in req]
    requirements = [req.split("\\")[0].split(":")[0].strip() for req in requirements]

setup(
    name='mamkit',
    version='0.1.0',
    author='Eleonora Mancini, Federico Ruggeri',
    author_email='e.mancini@unibo.it',
    description='A Comprehensive Multimodal Argument Mining Toolkit. ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lt-nlp-lab-unibo/mamkit',
    project_urls={
        'Bug Tracker': "https://github.com/lt-nlp-lab-unibo/mamkit/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    license='MIT',
    packages=[
        'mamkit',
        'mamkit.configs',
        'mamkit.data',
        'mamkit.models',
        'mamkit.modules',
        'mamkit.utility'
    ],
    install_requires=requirements,
    python_requires=">=3.7"
)
