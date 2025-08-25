from setuptools import setup, find_packages
from pathlib import Path

def load_requirements(fname="requirements.txt"):
    reqs = []
    for line in Path(fname).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='GPmix',
    version='0.1.3',
    author='Emmanuel Akeweje and Mimi Zhang',
    author_email='eakeweje@tcd.ie',
    description='GPmix is an ensemble clustering algorithm for functional data via random projections.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/EAkeweje/GPmix',
    packages=find_packages(exclude=["docs"]),
    install_requires=load_requirements(),
    license='MIT',
    license_files=('LICENSE',),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    include_package_data=True
)