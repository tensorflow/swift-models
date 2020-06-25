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
import TensorFlow

struct BatchSize: BenchmarkSetting {
  var value: Int
  init(_ value: Int) {
    self.value = value
  }
}

struct Length: BenchmarkSetting {
  var value: Int
  init(_ value: Int) {
    self.value = value
  }
}

struct Synthetic: BenchmarkSetting {
  var value: Bool
  init(_ value: Bool) {
    self.value = value
  }
}

struct Backend: BenchmarkSetting {
  var value: Value
  init(_ value: Value) {
    self.value = value
  }
  enum Value {
    case x10
    case eager
  }
}

extension BenchmarkSettings {
  var batchSize: Int? {
    return self[BatchSize.self]?.value
  }

  var length: Int? {
    return self[Length.self]?.value
  }

  var synthetic: Bool {
    if let value = self[Synthetic.self]?.value {
      return value
    } else {
      fatalError("Synthetic setting must have a default.")
    }
  }

  var backend: Backend.Value {
    if let value = self[Backend.self]?.value {
      return value
    } else {
      fatalError("Backend setting must have a default.")
    }
  }

  var device: Device {
    switch backend {
    case .eager: return Device.defaultTFEager
    case .x10: return Device.defaultXLA
    }
  }
}

let defaultSettings: [BenchmarkSetting] = [
  TimeUnit(.s),
  InverseTimeUnit(.s),
  Backend(.eager),
  Synthetic(false),
  Columns([
    "name",
    "avg_exp_per_second",
    "exp_per_second",
    "startup_time",
    "step_time_median",
    "step_time_min",
    "step_time_max",
    "wall_time",
    "iterations",
  ]),
]
