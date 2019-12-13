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
               root_data_dir=None):
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

  def training(self):
    """Runner-ccallable benchmark entry point for training benchmark."""
    result = run_swift_benchmark(name=self.benchmark_name, variety='training')
    self.report_benchmark(**result)

  def inference(self):
    """Runner-callable benchmark entry point for inference benchmark."""
    result = run_swift_benchmark(name=self.benchmark_name, variety='inference')
    self.report_benchmark(**result)


# This location assumes that are we are running within the S4TF's perfzero
# docker image. Perfzero automatically clones the swift-models project into the
# corresponding site-packages location.
cwd = '/workspace/perfzero/workspace/site-packages/swift-models/'


def extract_extras(settings):
  """Extract additional benchmark metadata.

  Returns a json-style object or None for additional
  benchmark metadata such as flags that can be useful for debugging.
  """

  return


def extract_metrics(result, variety):
  """Extract perfzero metrics based on the measurmenets.

  Extracts metrics such as number of examples per second,
  based on the the original raw timings.
  """

  timings = result['timings']
  example_count = result['exampleCount']

  # 50 percentile of a single iteration running time in seconds.
  timings_s = np.array(timings) / 1000
  wall_time = np.percentile(timings_s, 50)

  # Average examples per second across the entire
  # benchmark run. Doesn't account for warm-up.
  total_time_s = sum(timings_s)
  total_num_examples = example_count * len(timings_s)
  average_examples_per_second = total_num_examples / total_time_s

  # Examples per second across the second half
  # of the measurements. First half of measurements
  # is dropped to account for warm-up.
  warmup = int(len(timings_s) / 2 if len(timings_s) > 1 else 0)
  warm_timings_s = timings_s[warmup:]
  warm_time_s = sum(warm_timings_s)
  warm_num_examples = example_count * len(warm_timings_s)
  examples_per_second = warm_num_examples / warm_time_s

  metrics = [{
      'name': 'exp_per_second',
      'value': examples_per_second
  }, {
      'name': 'avg_exp_per_second',
      'value': average_examples_per_second
  }]

  return (wall_time, metrics)


def run_swift_benchmark(name, variety):
  print('running swift benchmark {} ({})'.format(name, variety))
  output = subp.check_output([
      'swift', 'run', '-c', 'release', 'Benchmarks', 'measure', '--benchmark',
      name, '--' + variety, '--json'
  ], cwd=cwd)
  result = json.loads(output)
  print('got json result back from swift: ')
  print(result)
  settings = result['configuration']['settings']
  wall_time, metrics = extract_metrics(result, variety)
  return {
      'iters': settings['iterations'],
      'wall_time': wall_time,
      'extras': extract_extras(settings),
      'metrics': metrics
  }


def new_swift_benchmark(name):
  """Create a new benchmark class with given name."""
  return type(name, (SwiftBenchmark,), {'benchmark_name': name})


def discover_swift_benchmarks():
  """Automatically discover and register swift benchmarks.

  Invokes swift benchmark cli to enumarate all available benchmarks
  and add them as new top-level types in the current module.
  """

  g = globals()
  defaults = subp.check_output([
      'swift', 'run', '-c', 'release', 'Benchmarks', 'list-defaults', '--json'
  ], cwd=cwd)
  for line in defaults.split(b'\n'):
    if len(line) > 0:
      name = json.loads(line)['name']
      if name not in g:
        g[name] = new_swift_benchmark(name)
        print("discovered swift benchmark '{}'".format(name))


discover_swift_benchmarks()
