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

public struct BatchSize: BenchmarkSetting {
  var value: Int
  init(_ value: Int) {
    self.value = value
  }
}

public struct Length: BenchmarkSetting {
  var value: Int
  init(_ value: Int) {
    self.value = value
  }
}

public struct Synthetic: BenchmarkSetting {
  var value: Bool
  init(_ value: Bool) {
    self.value = value
  }
}

public struct Backend: BenchmarkSetting {
  var value: Value
  init(_ value: Value) {
    self.value = value
  }
  public enum Value {
    case x10
    case eager
  }
}

public struct Platform: BenchmarkSetting {
  var value: Value
  init(_ value: Value) {
    self.value = value
  }
  public enum Value {
    case `default`
    case cpu
    case gpu
    case tpu
  }
}

public struct DatasetFilePath: BenchmarkSetting {
  var value: String
  init(_ value: String) {
    self.value = value
  }
}

extension BenchmarkSettings {
  public var batchSize: Int? {
    return self[BatchSize.self]?.value
  }

  public var length: Int? {
    return self[Length.self]?.value
  }

  public var synthetic: Bool {
    if let value = self[Synthetic.self]?.value {
      return value
    } else {
      fatalError("Synthetic setting must have a default.")
    }
  }

  public var backend: Backend.Value {
    if let value = self[Backend.self]?.value {
      return value
    } else {
      fatalError("Backend setting must have a default.")
    }
  }

  public var platform: Platform.Value {
    if let value = self[Platform.self]?.value {
      return value
    } else {
      fatalError("Platform setting must have a default.")
    }
  }

  public var device: Device {
    // Note: The line is needed, or all GPU memory
    // will be exhausted on initial allocation of the model.
    // TODO: Remove the following tensor workaround when above is fixed.
    let _ = _ExecutionContext.global

    switch backend {
    case .eager:
      switch platform {
      case .default: return Device.defaultTFEager
      case .cpu: return Device(kind: .CPU, ordinal: 0, backend: .TF_EAGER)
      case .gpu: return Device(kind: .GPU, ordinal: 0, backend: .TF_EAGER)
      case .tpu: fatalError("TFEager is unsupported on TPU.")
      }
    case .x10:
      switch platform {
      case .default: return Device.defaultXLA
      case .cpu: return Device(kind: .CPU, ordinal: 0, backend: .XLA)
      case .gpu: return Device(kind: .GPU, ordinal: 0, backend: .XLA)
      case .tpu: return (Device.allDevices.filter { $0.kind == .TPU }).first!
      }
    }
  }

  public var datasetFilePath: String? {
    return self[DatasetFilePath.self]?.value
  }
}

public let defaultSettings: [BenchmarkSetting] = [
  TimeUnit(.s),
  InverseTimeUnit(.s),
  Backend(.eager),
  Platform(.default),
  Synthetic(false),
  Columns([
    "name",
    "wall_time",
    "startup_time",
    "iterations",
    "avg_exp_per_second",
    "exp_per_second",
    "step_time_median",
    "step_time_min",
    "step_time_max",
  ]),
]
