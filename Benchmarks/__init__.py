# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np
import tensorflow as tf
import subprocess as subp


class SwiftBenchmark(tf.test.Benchmark):
  """Perfzero-compatible Swift benchmark."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               flag_methods=None,
               root_data_dir=None,
               tpu=None):
    """Perfzero-friendly constructor, we don't use most of those settings in swift."""
    if not output_dir:
      output_dir = '/tmp'
    self.output_dir = output_dir
    self.root_data_dir = root_data_dir
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Nothing to do here, but we need this for perfzero compat reasons."""
    pass


# This location assumes that are we are running within the S4TF's perfzero
# docker image. Perfzero automatically clones the swift-models project into the
# corresponding site-packages location.
cwd = '/workspace/benchmarks/perfzero/workspace/site-packages/swift-models'


def run_swift_benchmark(name):
  print('running swift benchmark {}'.format(name))
  # TODO: Remove the need for 2 warmup batches when we have better-shaped zero tangent vectors.
  output = subp.check_output([
      'swift', 'run', '-c', 'release', 'Benchmarks', 
      '--filter', name, '--warmup-iterations', '2', '--format', 'json',
      # Run each benchmark for up to 5 minutes.
      '--min-time', '300',  
  ], cwd=cwd)
  result = json.loads(output)
  print('got json result back from swift: ')
  print(result)

  wall_time = 0
  iterations = 0
  metrics = [] 
  for result in result["benchmarks"]:
    if result['name'] == name:
      for k, v in result.items():
        if k == 'name':
          pass
        elif k == 'wall_time':
          print("wall_time: {}".format(v))
          wall_time = v
        elif k == 'iterations':
          print("iterations: {}".format(v))
          iterations = int(v)
        else:
          print("metric: {}={}".format(k, v))
          metrics.append({
              'name': k,
              'value': v
          })

  result = {
      'iters': iterations,
      'wall_time': wall_time,
      'extras': None,
      'metrics': metrics
  }

  print("result: {}".format(result))

  return result


_benchmark_method_template = """
def {}(self):
  result = run_swift_benchmark(name="{}")
  self.report_benchmark(**result)
"""


def new_swift_benchmark_suite(suite, benchmarks):
  """Create a new benchmark suite with given name, and benchmarks."""

  print("creating benchmark suite: {}, {}".format(suite, benchmarks))

  def make_benchmark_method(suite, benchmark):
    print("making benchmark: {}, {}".format(suite, benchmark))

    exec(_benchmark_method_template.format(benchmark, "{}.{}".format(suite, benchmark)))

    return locals()[benchmark]

  methods = {}
  for benchmark in benchmarks:
    methods[benchmark] = make_benchmark_method(suite, benchmark)

  return type(suite, (SwiftBenchmark,), methods)


def discover_swift_benchmarks():
  """Automatically discover and register swift benchmarks.

  Invokes swift benchmark cli to enumarate all available benchmarks
  and add them as new top-level types in the current module.
  """

  output = subp.check_output([
      'swift', 'run', '-c', 'release', 'Benchmarks', 
      '--iterations', '0', '--warmup-iterations', '0',
      '--columns', 'name', '--format', 'json'
  ], cwd=cwd)

  suites = {}
  for benchmark in json.loads(output)['benchmarks']:
    name = benchmark['name']
    print("discovered swift benchmark '{}'".format(name))
    suite, benchmark = name.split(".")
    if suite not in suites:
      suites[suite] = []
    suites[suite].append(benchmark)

  g = globals()
  for suite, benchmarks in suites.items():
    g[suite] = new_swift_benchmark_suite(suite, benchmarks)


discover_swift_benchmarks()
