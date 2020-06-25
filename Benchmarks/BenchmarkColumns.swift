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

func median(_ arr: [Double]) -> Double {
  if arr.count == 0 { return 0 }
  if arr.count == 1 { return arr[0] }
  if arr.count == 2 { return (arr[0] + arr[1]) / 2 }

  // If we have odd number of elements, then
  // center element is the median.
  let s = arr.sorted()
  let center = arr.count / 2
  if arr.count % 2 == 1 {
    return s[center]
  }

  // If have even number of elements we need
  // to return an average between two middle elements.
  let center2 = arr.count / 2 - 1
  return (s[center] + s[center2]) / 2
}

func registerCustomColumns() {
  let timeFormatter: BenchmarkColumn.Formatter = { (value, settings) in
    "\(value) s"
  }
  let inverseTimeFormatter: BenchmarkColumn.Formatter = { (value, settings) in
    "\(value) /s"
  }
  func seconds(_ value: Double) -> Double {
    return value / 1000_000_000
  }
  BenchmarkColumn.registry["avg_exp_per_second"] =
    BenchmarkColumn(
      name: "avg_exp_per_second",
      value: { result in
        if let batchSize = result.settings.batchSize {
          let count = result.measurements.count
          let warmupCount = result.warmupMeasurements.count
          let examples = batchSize * (count + warmupCount)
          let time = result.measurements.reduce(0, +)
          let warmupTime = result.warmupMeasurements.reduce(0, +)
          return Double(examples) / seconds(time + warmupTime)
        } else {
          return 0
        }
      },
      alignment: .right,
      formatter: inverseTimeFormatter)
  BenchmarkColumn.registry["exp_per_second"] =
    BenchmarkColumn(
      name: "exp_per_second",
      value: { result in
        if let batchSize = result.settings.batchSize {
          let count = result.measurements.count
          let examples = batchSize * count
          let time = result.measurements.reduce(0, +)
          return Double(examples) / seconds(time)
        } else {
          return 0
        }
      },
      alignment: .right,
      formatter: inverseTimeFormatter)
  BenchmarkColumn.registry["startup_time"] =
    BenchmarkColumn(
      name: "startup_time",
      value: { seconds($0.warmupMeasurements.reduce(0, +)) },
      alignment: .right,
      formatter: timeFormatter)
  BenchmarkColumn.registry["step_time_median"] =
    BenchmarkColumn(
      name: "step_time_median",
      value: { seconds(median($0.measurements)) },
      alignment: .right,
      formatter: timeFormatter)
  BenchmarkColumn.registry["step_time_min"] =
    BenchmarkColumn(
      name: "step_time_min",
      value: { result in
        if let value = result.measurements.min() {
          return seconds(value)
        } else {
          return 0
        }
      },
      alignment: .right,
      formatter: timeFormatter)
  BenchmarkColumn.registry["step_time_max"] =
    BenchmarkColumn(
      name: "step_time_max",
      value: { result in
        if let value = result.measurements.max() {
          return seconds(value)
        } else {
          return 0
        }
      },
      alignment: .right,
      formatter: timeFormatter)
  BenchmarkColumn.registry["wall_time"] =
    BenchmarkColumn(
      name: "wall_time",
      value: { seconds($0.measurements.reduce(0, +) + $0.warmupMeasurements.reduce(0, +)) },
      alignment: .right,
      formatter: timeFormatter)
}
