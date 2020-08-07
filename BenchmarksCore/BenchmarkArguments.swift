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

import ArgumentParser
import Benchmark

public struct BenchmarkArguments: ParsableArguments {
  @OptionGroup()
  var arguments: Benchmark.BenchmarkArguments

  @Option(help: "Size of a single batch.")
  var batchSize: Int?

  @Flag(help: "Use eager backend.")
  var eager: Bool = false

  @Flag(help: "Use X10 backend.")
  var x10: Bool = false

  @Flag(help: "Use CPU platform.")
  var cpu: Bool = false

  @Flag(help: "Use GPU platform.")
  var gpu: Bool = false

  @Flag(help: "Use TPU platform.")
  var tpu: Bool = false

  @Flag(help: "Use synthetic data.")
  var synthetic: Bool = false 

  @Flag(help: "Use real data.")
  var real: Bool = false 

  @Option(help: "File path for dataset loading.")
  var datasetFilePath: String?

  public init() {}

  public init(arguments: Benchmark.BenchmarkArguments, batchSize: Int?, eager: Bool, x10: Bool,
              cpu: Bool, gpu: Bool, tpu: Bool, synthetic: Bool, real: Bool,
              datasetFilePath: String?) {
    self.arguments = arguments
    self.batchSize = batchSize
    self.eager = eager
    self.x10 = x10
    self.cpu = cpu
    self.gpu = gpu
    self.tpu = tpu
    self.synthetic = synthetic
    self.real = real
    self.datasetFilePath = datasetFilePath
  }

  public mutating func validate() throws {
    try arguments.validate()

    guard !(real && synthetic) else {
      throw ValidationError(
        "Can't specify both --real and --synthetic data sources.")
    }

    guard !(eager && x10) else {
      throw ValidationError(
        "Can't specify both --eager and --x10 backends.")
    }

    guard !(cpu && gpu) || !(cpu && tpu) || !(gpu && tpu) else {
      throw ValidationError(
        "Can't specify multiple platforms.")
    }
  }

  public var settings: [BenchmarkSetting] {
    var settings = arguments.settings

    if let value = batchSize {
      settings.append(BatchSize(value))
    }
    if x10 {
      settings.append(Backend(.x10))
    }
    if eager {
      settings.append(Backend(.eager))
    }
    if cpu {
      settings.append(Platform(.cpu))
    }
    if gpu {
      settings.append(Platform(.gpu))
    }
    if tpu {
      settings.append(Platform(.tpu))
    }
    if synthetic {
      settings.append(Synthetic(true))
    }
    if real {
      settings.append(Synthetic(false))
    }
    if let value = datasetFilePath {
      settings.append(DatasetFilePath(value))
    }

    return settings
  }
}
