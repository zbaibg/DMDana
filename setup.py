try:
    import git
except ImportError as e:
    raise ImportError(f"GitPython is not installed. Please install it before installing this package. Error details: {e}")
from setuptools import find_packages, setup


def get_git_revision_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='DMDana',
    version='0.1.0+' + get_git_revision_hash(),
    packages=find_packages(),
    install_requires=required,
    package_data={'': ['DMDana_default.ini','githash.log']},
    include_package_data=True,
    python_requires='>3.2'
)