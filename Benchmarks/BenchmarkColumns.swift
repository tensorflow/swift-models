// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Benchmark

func registerCustomColumns() {
  BenchmarkColumn.register(
    BenchmarkColumn(
      name: "avg_exp_per_second",
      value: { result in
        if let batchSize = result.settings.batchSize {
          let count = result.measurements.count
          let warmupCount = result.warmupMeasurements.count
          let examples = batchSize * (count + warmupCount)
          let time = result.measurements.reduce(0, +)
          let warmupTime = result.warmupMeasurements.reduce(0, +)
          return Double(examples) / (time + warmupTime)
        } else {
          return 0
        }
      },
      unit: .inverseTime))
  BenchmarkColumn.register(
    BenchmarkColumn(
      name: "exp_per_second",
      value: { result in
        if let batchSize = result.settings.batchSize {
          let count = result.measurements.count
          let examples = batchSize * count
          let time = result.measurements.reduce(0, +)
          return Double(examples) / time
        } else {
          return 0
        }
      },
      unit: .inverseTime))
  BenchmarkColumn.register(
    BenchmarkColumn.registry["warmup"]!.renamed("startup_time"))
  BenchmarkColumn.register(
    BenchmarkColumn.registry["median"]!.renamed("step_time_median"))
  BenchmarkColumn.register(
    BenchmarkColumn.registry["min"]!.renamed("step_time_min"))
  BenchmarkColumn.register(
    BenchmarkColumn.registry["max"]!.renamed("step_time_max"))
  BenchmarkColumn.register(
    BenchmarkColumn(
      name: "wall_time",
      value: { $0.measurements.reduce(0, +) + $0.warmupMeasurements.reduce(0, +) },
      unit: .time))
}
