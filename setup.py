#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""setup package."""
import os
import stat
import platform
import sys

from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

version = '2.1.0'

backend_policy = os.getenv('BACKEND_POLICY')

pwd = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(pwd, 'build/package')


def _read_file(filename):
    with open(os.path.join(pwd, filename), encoding='UTF-8') as f:
        return f.read()


def get_platform():
    """
    Get platform name.

    Returns:
        str, platform name in lowercase.
    """
    return platform.system().strip().lower()


readme = _read_file('README.md')


def _write_version(file):
    file.write("__version__ = '{}'\n".format(version))


def _write_config(file):
    file.write("__backend__ = '{}'\n".format(backend_policy))


def build_dependencies():
    """generate python file"""
    version_file = os.path.join(pkg_dir, 'mindspore_serving', 'version.py')
    with open(version_file, 'w') as f:
        _write_version(f)

    version_file = os.path.join(pwd, 'mindspore_serving', 'version.py')
    with open(version_file, 'w') as f:
        _write_version(f)

    config_file = os.path.join(pkg_dir, 'mindspore_serving', 'default_config.py')
    with open(config_file, 'w') as f:
        _write_config(f)

    config_file = os.path.join(pwd, 'mindspore_serving', 'default_config.py')
    with open(config_file, 'w') as f:
        _write_config(f)


required_package = [
    'numpy>=1.21.6',
    'protobuf >= 3.13.0',
    'psutil >= 5.9.1',
    'sse-starlette',
    'sseclient',
    'easydict',
    'fastapi',
    'uvicorn',
    'sentencepiece',
    'transformers==4.35.0'
]

package_data = {
    '': [
        '*.so*',
        '*.pyd',
        '*.dll',
        'lib/*.so*',
        'lib/*.a',
        '_mindspore_serving',
        'proto/*.py'
    ]
}


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    if platform.system() == "Windows":
        return

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE |
                     stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def bin_files():
    """
    Gets the binary files to be installed.
    """
    data_files = []
    binary_files = []

    cache_server_bin = os.path.join('mindspore_serving', 'bin', 'cache_server')
    if not os.path.exists(cache_server_bin):
        return data_files
    binary_files.append(cache_server_bin)
    cache_admin_bin = os.path.join('mindspore_serving', 'bin', 'cache_admin')
    if not os.path.exists(cache_admin_bin):
        return data_files
    binary_files.append(cache_admin_bin)
    data_files.append(('bin', binary_files))
    return data_files


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        super().run()
        egg_info_dir = os.path.join(pkg_dir, 'mindspore_serving.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""

    def run(self):
        super().run()
        mindspore_dir = os.path.join(pkg_dir, 'build', 'mindspore_serving/lib', 'mindspore_serving')
        update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pkg_dir, 'build', 'mindspore_serving/lib', 'akg')
        update_permissions(mindspore_dir)


if __name__ == '__main__':
    setup(
        name="mindspore_serving",
        version=version,
        author='The MindSpore Authors',
        author_email='contact@mindspore.cn',
        url='https://www.mindspore.cn',
        download_url='https://gitee.com/mindspore/serving/tags',
        project_urls={
            'Sources': 'https://gitee.com/mindspore/serving',
            'Issue Tracker': 'https://gitee.com/mindspore/serving/issues',
        },
        description='MindSpore is a new open source deep learning training/inference '
                    'framework that could be used for mobile, edge and cloud scenarios.',
        long_description="\n\n".join([readme]),
        long_description_content_type="text/markdown",
        data_files=bin_files(),
        packages=find_packages(),
        package_data=package_data,
        platforms=[get_platform()],
        include_package_data=True,
        cmdclass={
            'egg_info': EggInfo,
            'build_py': BuildPy,
        },
        python_requires='>=3.9.9',
        install_requires=required_package,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='mindspore machine learning',
    )
