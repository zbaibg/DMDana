import subprocess
from setuptools import find_packages, setup

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except FileNotFoundError:
        raise EnvironmentError("git is not installed on this system.")

try:
    # Attempt to get the git hash to see if git is available
    git_revision_hash = get_git_revision_hash()
except EnvironmentError as e:
    raise ImportError(f"Unable to find git. Please ensure that git is installed and available in your PATH. Error details: {e}")

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='DMDana',
    version='0.1.0+' + git_revision_hash(),
    packages=find_packages(),
    install_requires=required,
    package_data={'': ['DMDana_default.ini','githash.log']},
    include_package_data=True,
    python_requires='>3.2'
)