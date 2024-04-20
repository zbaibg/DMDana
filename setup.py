import git
from setuptools import find_packages, setup

repo = git.Repo('.',search_parent_directories=True)
sha = repo.head.object.hexsha
with open('./DMDana/githash.log','w') as file:
    file.write(sha)
    print(sha,file=file)
    print('This file is only updated when using pip install.',file=file)
    print('The code would first check and use the hash from the .git folder in the parent folder if it exsits (for develop mode).',file=file)
    print('If there is no .git folder in the parent folder, the code would check this githash.log file (for installed mode)',file=file)
setup(
    name='DMDana',
    version='1.0.0',
    packages=find_packages(),
    package_data={'': ['DMDana_default.ini','githash.log']},
    include_package_data=True,
    install_requires=[],
)