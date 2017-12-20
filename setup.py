import os

from setuptools import setup, find_packages

import quantflow as qf


def read(name):
    filename = os.path.join(os.path.dirname(__file__), name)
    with open(filename) as fp:
        return fp.read()


def requirements(name):
    install_requires = []
    dependency_links = []

    for line in read(name).split('\n'):
        if line.startswith('-e '):
            link = line[3:].strip()
            if link == '.':
                continue
            dependency_links.append(link)
            line = link.split('=')[1]
        line = line.strip()
        if line:
            install_requires.append(line)

    return install_requires, dependency_links


meta = dict(
    version=qf.__version__,
    description=qf.__doc__,
    name='quantflow',
    author="Luca Sbardella",
    author_email="luca@quantmind.com",
    url="https://github.com/quantmind/quantflow",
    zip_safe=False,
    license='BSD',
    long_description=read('readme.md'),
    setup_requires=['pulsar', 'wheel'],
    install_requires=requirements('requirements/hard.txt')[0],
    packages=find_packages(include=['quantflow', 'quantflow.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Plugins',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities']
)


if __name__ == '__main__':
    setup(**meta)
