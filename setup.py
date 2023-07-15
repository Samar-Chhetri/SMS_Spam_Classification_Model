from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements




setup(
    name = 'SMS_Spam_Classification_Model',
    version='0.0.1',
    author='Samar Chhetri',
    author_email='samarchhetri23@gmail.com',
    packages=find_packages(),
    long_description='An application which classify an email as spam or not.',
    install_requires = get_requirements('requirements.txt')
)