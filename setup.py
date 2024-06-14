from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name='MY_Project_work',
    version='0.0.1',
    author='Sagar Kasotiya',
    author_email='sagarkansotiya123@gmail.com',
    install_requires= get_requirements('requirements.txt'),
    packages=find_packages()
)