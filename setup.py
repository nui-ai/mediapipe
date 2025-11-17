"""Copyright 2020-2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

==================================================

Setup for the MediaPipe package via setuptools api,
this setup.py is used by pip to build a python package (as a wheel)
which uses the defined bazel build sources providing the underlying necessary C++ code and its accompanying artefact files like tflite modules.
the python package's requirements recorded into the built wheel are taken by this script from the requirements.txt file.

Notes:

1. Modified in various ways to make it work for this v0.10.13 code version at the current time, see the commit log and readme.
2. All attempts to make it make bazel only build incrementally initially failed (see https://chatgpt.com/c/68ce82f1-d284-8327-90a0-e4980994cf35)
   but as we currently use a global bazel cache in our bazel build commands, that is no longer the case, i.e. it will reuse from there.
3. Modern pip swallows all bazel stdout/stderr output, unless you attach -v in the pip command, so using -v is highly recommended for
   orientation and traceability. """

import os
import glob
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# setuptools is automatically installed by pip when it runs this script,
# and may be flagged as unknown symbols by the IDE if setuptools is not
# also installed in the project's active python environment.
import setuptools
from setuptools.command import build_ext
from setuptools.command import build_py
from setuptools.command import install

# print the value of the MEDIAPIPE_PYTHON_BIN env variable, which is used by pip do determine the target python env to install to,
# which was found to be insofar the most reliable way to accomplish that which is also simple.
print(f'target python environment: {os.environ.get("MEDIAPIPE_PYTHON_BIN")}', flush=True)

git_hash = subprocess.check_output(
    ['git', 'rev-parse', '--short', 'HEAD'],
    cwd = os.path.dirname(os.path.abspath(__file__))
    ).decode('utf-8').strip()

__version__ = f"0+{git_hash}"


IS_WINDOWS = (platform.system() == 'Windows')
IS_MAC = (platform.system() == 'Darwin')
MP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MP_DIR_INIT_PY = os.path.join(MP_ROOT_PATH, 'mediapipe/__init__.py')
MP_THIRD_PARTY_BUILD = os.path.join(MP_ROOT_PATH, 'third_party/BUILD')
MP_ROOT_INIT_PY = os.path.join(MP_ROOT_PATH, '__init__.py')


# take in the environment variable affecting whether to build for the GPU code path too
MP_DISABLE_GPU = os.environ.get('MEDIAPIPE_DISABLE_GPU') != '0'
GPU_OPTIONS_DISBALED = ['--define=MEDIAPIPE_DISABLE_GPU=1']
GPU_OPTIONS_ENBALED = [
    '--copt=-DTFLITE_GPU_EXTRA_GLES_DEPS',
    '--copt=-DMEDIAPIPE_OMIT_EGL_WINDOW_BIT',
    '--copt=-DMESA_EGL_NO_X11_HEADERS',
    '--copt=-DEGL_NO_X11',
]
if IS_MAC:
  GPU_OPTIONS_ENBALED.append('--copt=-DMEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER')
GPU_OPTIONS = GPU_OPTIONS_DISBALED if MP_DISABLE_GPU else GPU_OPTIONS_ENBALED


def _normalize_path(path):
  return path.replace('\\', '/') if IS_WINDOWS else path


def _get_backup_file(path):
  return path + '.backup'


def _parse_requirements(path):
  with open(os.path.join(MP_ROOT_PATH, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


def _get_long_description():
  # Fix the image urls.
  return re.sub(
      r'(docs/images/|docs/images/mobile/)([A-Za-z0-9_]*\.(png|gif))',
      r'https://github.com/google/mediapipe/blob/master/\g<1>\g<2>?raw=true',
      open(os.path.join(MP_ROOT_PATH, 'README.md'),
           'rb').read().decode('utf-8'))


def _check_bazel():
  """Check Bazel binary as well as its version."""

  if not shutil.which('bazel'):
    sys.stderr.write('could not find a bazel executable.')
    sys.exit(-1)
  try:
    bazel_version_info = subprocess.check_output(['bazel', '--version'])
  except subprocess.CalledProcessError as e:
    sys.stderr.write('fail to get bazel version by $ bazel --version: ' + str(e.output))
    sys.exit(-1)

  print(f'using bazel version: {bazel_version_info.decode("UTF-8")}', flush=True)

  bazel_version_info = bazel_version_info.decode('UTF-8').strip()
  version = bazel_version_info.split('bazel ')[1].split('-')[0]
  version_segments = version.split('.')
  # Treat "0.24" as "0.24.0"
  if len(version_segments) == 2:
    version_segments.append('0')
  for seg in version_segments:
    if not seg.isdigit():
      sys.stderr.write('invalid bazel version number: %s\n' % version_segments)
      sys.exit(-1)
  bazel_version = int(''.join(['%03d' % int(seg) for seg in version_segments]))
  if bazel_version < 3004000:
    sys.stderr.write(
        'the current bazel version is older than the minimum version that MediaPipe can support. Please upgrade bazel.'
    )
    sys.exit(-1)


def _write__init__source_file():
    """ writes mediapipe/__init__.py rather than trust its git contents, we could throw away this piece if we wanted and just pin and trust the contents of mediapipe/__init__.py in git.
        basically this content exports a subset of the mediapipe submodules, and then deletes some of what's been exported so a little messy to simplify. """

    content_to_write = [
        '# Copyright 2019 - 2022 The MediaPipe Authors.\n',
        '#\n',
        '# Licensed under the Apache License, Version 2.0 (the "License");\n',
        '# you may not use this file except in compliance with the License.\n',
        '# You may obtain a copy of the License at\n',
        '#\n',
        '#      http://www.apache.org/licenses/LICENSE-2.0\n',
        '#\n',
        '# Unless required by applicable law or agreed to in writing, software\n',
        '# distributed under the License is distributed on an "AS IS" BASIS,\n',
        '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n',
        '# See the License for the specific language governing permissions and\n',
        '# limitations under the License.\n',
        '\n',
        '# NOTE: this file is currently dynamically generated by pip installing (see setup.py) and thus may not be the original git borne version of itself.\n',
        '\n',
        'from mediapipe.python import *\n',
        'import mediapipe.python.solutions as solutions \n',
        'import mediapipe.tasks.python as tasks\n',
        '\n',
        'del framework\n',
        'del gpu\n',
        'del modules\n',
        'del python\n',
        'del mediapipe\n',
        'del util\n',
        f"__version__ = '{__version__}'\n"
    ]
    expected_content = ''.join(content_to_write)

    current_content = None
    if os.path.exists(MP_DIR_INIT_PY):
        with open(MP_DIR_INIT_PY, 'r') as f:
            current_content = f.read()
    if current_content == expected_content:
        print(f"{MP_DIR_INIT_PY} already up-to-date, not modified.", flush=True)
        return

    with open(MP_DIR_INIT_PY, 'w') as mp_dir_init_file:
        mp_dir_init_file.writelines(content_to_write)
    print(f"generated {MP_DIR_INIT_PY}.", flush=True)


def _copy_to_build_lib_dir(build_lib, file):
  """Copy a file from bazel-bin to the build lib dir."""
  dst = os.path.join(build_lib + '/', file)
  dst_dir = os.path.dirname(dst)
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
  shutil.copyfile(os.path.join('bazel-bin/', file), dst)


def _invoke_shell_command(shell_commands):
  """Invokes shell command from the list of arguments."""
  print('Invoking shell command:', shlex.join(shell_commands), flush=True)
  try:
    subprocess.run(shell_commands, check=True)
  except subprocess.CalledProcessError as e:
    print(e)
    sys.exit(e.returncode)


def _build_bazel_command(target, extra_bazel_args=None):
    """ Constructs a consistent Bazel build command for all build invocations.
    This avoids Bazel discarding the analysis cache due to option changes between the different bazel build invocations made
    by the current source file, as otherwise seen in messages like: "INFO: Build options --action_env, --compilation_mode, --copt, and 1 more have changed, discarding analysis cache"
    which it would if the build command is invoked with different options each time this script invokes it, in which case bazel would
    otherwise rebuild everything from scratch on every bazel command rather than use its analysis cache to determine what is already
    up-to-date and what needs to be rebuilt when starting a bazel build command. since this script invokes bazel commands several
    times this is necessary. """

    # the bazel build flags/options herea are currently mimicking the ones used by the underlying C++ bazel project
    # for all of its targets, not just its C++ targets used by the python mediapipe package being built by pip.
    bazel_command = [
        'bazel',
        'build',
        '-c', 'opt',
        '--define', 'MEDIAPIPE_DISABLE_GPU=1',
        '--define', 'OPENCV=source',
        # not sure whether it uses this path, in the python pip context of an isolated pip build
        '--disk_cache=/home/matan/.cache/bazel-disk-cache',
        # enables debugging of the c++ code (somewhat less relevant in the python package context, yet ..)
        '--fission=no',
        # '--nostamp',
        target,
    ]
    return bazel_command


class GeneratePyProtos(build_ext.build_ext):
  """Generates MediaPipe Python protobuf files through the Protocol Buffers Compiler."""
  def run(self):
    if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
      self._protoc = os.environ['PROTOC']
    else:
      self._protoc = shutil.which('protoc')
    if self._protoc is None:
      sys.stderr.write(
          'protoc is not found. Please run \'apt install -y protobuf'
          '-compiler\' (linux) or \'brew install protobuf\'(macos) to install '
          'protobuf compiler binary.')
      sys.exit(-1)

    # adding __init__.py to the mediapipe proto generated directories
    proto_dirs = ['mediapipe/calculators'] + [
        x[0] for x in os.walk('mediapipe/modules')
    ] + [x[0] for x in os.walk('mediapipe/tasks/cc')]
    for proto_dir in proto_dirs:
      self._add_empty_init_file(
          os.path.abspath(
              os.path.join(MP_ROOT_PATH, self.build_lib, proto_dir, '__init__.py')))

    # proto files to generate from for using the mediapipe framework
    for pattern in [
        'mediapipe/framework/**/*.proto', 'mediapipe/calculators/**/*.proto',
        'mediapipe/gpu/**/*.proto', 'mediapipe/modules/**/*.proto',
        'mediapipe/tasks/cc/**/*.proto', 'mediapipe/util/**/*.proto',
        'mediapipe/examples/desktop/**/pipeline_output.proto'  # piggyback the generation of a python class from our own non-mediapipe protobuf type here
    ]:
      for proto_file in glob.glob(pattern, recursive=True):
        # Ignore test protos.
        if proto_file.endswith('test.proto'):
          continue
        # Ignore tensorflow protos in mediapipe/calculators/tensorflow.
        if 'tensorflow' in proto_file:
          continue
        # Ignore testdata dir.
        if 'testdata' in proto_file:
          continue
        self._add_empty_init_file(
            os.path.abspath(
                os.path.join(MP_ROOT_PATH, self.build_lib, os.path.dirname(proto_file), '__init__.py')))
        self._generate_proto(proto_file)

  def _add_empty_init_file(self, init_file):
    init_py_dir = os.path.dirname(init_file)
    if not os.path.exists(init_py_dir):
      os.makedirs(init_py_dir)
    if not os.path.exists(init_file):
      open(init_file, 'w').close()

  def _generate_proto(self, proto_source_file):
    """Invokes the Protocol Compiler to generate a _pb2.py."""
    generated_source_file = proto_source_file.replace('.proto', '_pb2.py')
    output = os.path.join(self.build_lib, generated_source_file)
    sys.stderr.write(f'generating a python source file from proto source file {proto_source_file}: %s\n' % output)
    protoc_command = [self._protoc, '-I.', '--python_out=' + os.path.abspath(self.build_lib), proto_source_file]
    _invoke_shell_command(protoc_command)


class BuildModules(build_ext.build_ext):
  """Build binary graphs and download external files of various MediaPipe modules."""

  user_options = build_ext.build_ext.user_options + [
      ('link-opencv', None, 'if true, build opencv from source.'),
  ]
  boolean_options = build_ext.build_ext.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_ext.build_ext.initialize_options(self)

  def finalize_options(self):
    build_ext.build_ext.finalize_options(self)

  def run(self):
    _check_bazel()
    external_files = [
        'hand_landmark/hand_landmark_full.tflite',
        'palm_detection/palm_detection_full.tflite',
    ]
    for elem in external_files:
      external_file = os.path.join('mediapipe/modules/', elem)
      sys.stderr.write('downloading file: %s\n' % external_file)
      self._download_external_file(external_file)

    binary_graphs = [
        # we still have a necessary bazel build target (building a binary of our pipeline definition file) named like below, so this line is still correct and required for the python api.
        # this is required because unlike our C mains, our python implementation in Gesture Studio passes the binary form of the pipeline definition to the mediapipe api, though it may
        # equally pass the text form like our C++ mains do. if we switch to pass the text form, we no longer need to build a binary pb of it, but this doesn't really matter.
        'hand_landmark/hand_landmark_tracking_cpu.binarypb',
    ]
    for elem in binary_graphs:
      binary_graph = os.path.join('mediapipe/modules/', elem)
      self._build_mediapipe_graph_target(binary_graph)

  def _download_external_file(self, external_file):
        """Download an external file from GCS via Bazel."""
        fetch_model_command = _build_bazel_command(external_file)
        _invoke_shell_command(fetch_model_command)
        _copy_to_build_lib_dir(self.build_lib, external_file)

  def _build_mediapipe_graph_target(self, binary_graph_target):
    """Generate binary graph for a particular MediaPipe binary graph target."""

    bazel_command = _build_bazel_command(binary_graph_target)

    print(f'\nBuilding {binary_graph_target} ... the bazel command being issued for it follows.', flush=True)
    _invoke_shell_command(bazel_command)
    _copy_to_build_lib_dir(self.build_lib, binary_graph_target)


class GenerateMetadataSchema(build_ext.build_ext):
  """Generate metadata python schema files. This steps does not use bazel's build cache, it rebuilds every time from scratch. """

  user_options = build_ext.build_ext.user_options + [
      ('link-opencv', None, 'if true, build opencv from source.'),
  ]
  boolean_options = build_ext.build_ext.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_ext.build_ext.initialize_options(self)

  def run(self):
    for target in [
        'metadata_schema_py',
        'schema_py',
    ]:

      bazel_target = f'//mediapipe/tasks/metadata:{target}'
      bazel_command = _build_bazel_command(bazel_target)

      print(f'\nBuilding {target} ... the bazel command being issued for it follows.', flush=True)
      _invoke_shell_command(bazel_command)
      _copy_to_build_lib_dir(
          self.build_lib,
          'mediapipe/tasks/metadata/' + target + '_generated.py')
    for schema_file in [
        'mediapipe/tasks/metadata/metadata_schema.fbs',
        # 'mediapipe/tasks/metadata/object_detector_metadata_sbazechema.fbs',
        # 'mediapipe/tasks/metadata/image_segmenter_metadata_schema.fbs',
    ]:
      shutil.copyfile(schema_file,
                      os.path.join(self.build_lib + '/', schema_file))


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target, target_name=''):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        posixpath.relpath(bazel_target, '//').split(':'))
    if target_name:
      self.target_name = target_name
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
    setuptools.Extension.__init__(self, ext_name, sources=[])


class BuildExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  user_options = build_ext.build_ext.user_options + [
      ('link-opencv', None, 'if true, build opencv from source.'),
  ]
  boolean_options = build_ext.build_ext.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_ext.build_ext.initialize_options(self)

  def finalize_options(self):
    build_ext.build_ext.finalize_options(self)

  def run(self):
    _check_bazel()
    if IS_MAC:
      for ext in self.extensions:
        target_name = self.get_ext_fullpath(ext.name)
        # Build x86
        self._build_binary(
            ext,
            ['--cpu=darwin', '--ios_multi_cpus=i386,x86_64,armv7,arm64'],
        )
        x86_name = self.get_ext_fullpath(ext.name)
        # Build Arm64
        ext.name = ext.name + '.arm64'
        self._build_binary(
            ext,
            ['--cpu=darwin_arm64', '--ios_multi_cpus=i386,x86_64,armv7,arm64'],
        )
        arm64_name = self.get_ext_fullpath(ext.name)
        # Merge architectures
        lipo_command = [
            'lipo',
            '-create',
            '-output',
            target_name,
            x86_name,
            arm64_name,
        ]
        _invoke_shell_command(lipo_command)
        # Delete the arm64 file (the x86 file was overwritten by lipo)
        _invoke_shell_command(['rm', arm64_name])
    else:
      for ext in self.extensions:
        self._build_binary(ext)
    # Use self.build_lib as the destination for .so files
    _copy_opencv_libs(self.build_lib)
    build_ext.build_ext.run(self)

  def _build_binary(self, ext, extra_args=None):
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    bazel_command = _build_bazel_command(str(ext.bazel_target + '.so'), extra_args)
    print(f'\nBuilding {ext.bazel_target} ... the bazel command being issued for it follows.', flush=True)
    _invoke_shell_command(bazel_command)
    ext_bazel_bin_path = os.path.join('bazel-bin', ext.relpath,
                                      ext.target_name + '.so')
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)
    if IS_WINDOWS:
      for opencv_dll in glob.glob(
          os.path.join('bazel-bin', ext.relpath, '*opencv*.dll')):
        shutil.copy(opencv_dll, ext_dest_dir)


def _copy_opencv_libs(build_lib=None):
    """Copy Bazel-built OpenCV .so files into the build_lib/python/ for runtime loading, and set owner write permissions
    to support repeat `pip install .` runs being able to overwrite them."""
    opencv_lib_dir = os.path.join('bazel-bin', 'third_party', 'opencv_cmake', 'lib')
    # If build_lib is not provided, fallback to source tree (for legacy usage)
    if build_lib is None:
        dest_dir = os.path.join(MP_ROOT_PATH, 'mediapipe', 'python')
    else:
        dest_dir = os.path.join(build_lib, 'mediapipe', 'python')
    if not os.path.exists(opencv_lib_dir):
        print(f"OpenCV lib dir not found: {opencv_lib_dir}", flush=True)
        return
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for so_file in glob.glob(os.path.join(opencv_lib_dir, 'libopencv*.so*')):
        dest_file = os.path.join(dest_dir, os.path.basename(so_file))
        print(f"Copying {so_file} to {dest_dir}", flush=True)
        shutil.copy(so_file, dest_dir)
        try:
            os.chmod(dest_file, 0o744)  # Owner: rwx, Group: r, Others: r
        except Exception as e:
            print(f"Warning: could not set permissions on {dest_file}: {e}", flush=True)


class BuildPy(build_py.build_py):

  user_options = build_py.build_py.user_options + [
      ('link-opencv', None, 'if true, use the installed opencv library.'),
  ]
  boolean_options = build_py.build_py.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_py.build_py.initialize_options(self)

  def finalize_options(self):
    build_py.build_py.finalize_options(self)

  def run(self):
    _write__init__source_file()
    self.run_command('gen_protos')
    self.run_command('generate_metadata_schema')
    self.run_command('build_modules')
    self.run_command('build_ext')
    build_py.build_py.run(self)
    self.run_command('restore')


class Install(install.install):

  user_options = install.install.user_options + [
      ('link-opencv', None, 'if true, use the installed opencv library.'),
  ]
  boolean_options = install.install.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    install.install.initialize_options(self)

  def finalize_options(self):
    install.install.finalize_options(self)

  def run(self):
    self.run_command('build_py')
    install.install.run(self)


class Restore(setuptools.Command):
  """Restore the modified mediapipe source files."""

  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    return


setuptools.setup(
    name='hand-tracking-mp-lean',  # this is the package name, not what python code needs to import to use its content (which is still "mediapipe").
    version=__version__,
    url='https://github.com/nui-ai/mediapipe/tree/liberation',
    description="python api for a one-to-one results-compatible and performance equivalent hands tracking pipeline and no-pipeline redux of v.0.10.13's original hand tracking pipeline for desktop CPU",
    author='matan@nui.ai',
    author_email='matan@nui.ai',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(  # it would be nice to specify a list of which path's to include instead, if pruning was tractable by some cleaning workflow
        exclude=['mediapipe.examples.desktop.*',
                 'mediapipe.model_maker.*',
                 'mediapipe_analysis', 'mediapipe_analysis.*']),
    install_requires=_parse_requirements('requirements.txt'),
    cmdclass={
        'build_py': BuildPy,
        'build_modules': BuildModules,
        'build_ext': BuildExtension,
        'generate_metadata_schema': GenerateMetadataSchema,
        'gen_protos': GeneratePyProtos,
        'install': Install,
        'restore': Restore,
    },
    ext_modules=[ # these are lib files (.so on linux) that will be built through bazel, which the python code borne by the package will be using
        BazelExtension('//mediapipe/python:_framework_bindings'),
        BazelExtension('//mediapipe/tasks/cc/metadata/python:_pywrap_metadata_version'),
        BazelExtension('//mediapipe/tasks/python/metadata/flatbuffers_lib:_pywrap_flatbuffers'),
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3 :: Only',
    ],
    license='Apache 2.0',
    keywords='mediapipe',
)
