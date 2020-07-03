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
import Datasets
import ImageClassificationModels
import TensorFlow

func makeRandomTensor(
  batchSize: Int,
  dimensions: [Int],
  device: Device
) -> Tensor<Float> {
  var allDimensions = [batchSize]
  allDimensions.append(contentsOf: dimensions)
  let tensor = Tensor<Float>(
    randomNormal: TensorShape(allDimensions), mean: Tensor<Float>(0.5, on: device),
    standardDeviation: Tensor<Float>(0.1, on: device), seed: (0xffeffe, 0xfffe),
    on: device)
  return tensor
}

func makeForwardBenchmark<CustomLayer>(
  layer: CustomLayer.Type,
  inputDimensions: [Int],
  outputDimensions: [Int]
) -> ((inout BenchmarkState) throws -> Void)
where
  CustomLayer: Layer,
  CustomLayer: DefaultInit,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var layer = CustomLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize: batchSize,
      dimensions: inputDimensions,
      device: device)

    var sink: TensorShape = TensorShape([])

    while true {
      do {
        try state.measure {
          let result = layer(input)
          // Force materialization of the lazy results.
          sink = result.shape
          LazyTensorBarrier()
        }
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }

    // Control-flow never gets here, but this removes the warning 
    // about shape being never used.
    // being never used.
    fatalError("unrechable \(sink)")
  }
}

func makeGradientBenchmark<CustomLayer>(
  layer: CustomLayer.Type,
  inputDimensions: [Int],
  outputDimensions: [Int]
) -> ((inout BenchmarkState) throws -> Void)
where
  CustomLayer: Layer,
  CustomLayer: DefaultInit,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var layer = CustomLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize: batchSize,
      dimensions: inputDimensions,
      device: device)
    let output = makeRandomTensor(
      batchSize: batchSize,
      dimensions: outputDimensions,
      device: device)

    var sink: CustomLayer.TangentVector = CustomLayer.TangentVector.zero
    sink.move(to: device)

    while true {
      do {
        try state.measure {
          let result = TensorFlow.gradient(at: layer) { layer -> Tensor<Float> in
            let predicted = layer(input)
            return meanAbsoluteError(predicted: predicted, expected: output)
          }
          // Force materialization of the lazy results.
          sink += result
          LazyTensorBarrier()
        }
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }

    // Control-flow never gets here, but this removes the warning 
    // about shape being never used.
    // being never used.
    fatalError("unrechable \(sink)")
  }
}

func makeLayerSuite<CustomLayer>(
  layer: CustomLayer.Type,
  inputDimensions inp: [Int],
  outputDimensions outp: [Int]
) -> BenchmarkSuite
where
  CustomLayer: Layer,
  CustomLayer: DefaultInit,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  let name: String = String(String(reflecting: layer).split(separator: ".").last!)

  return BenchmarkSuite(
    name: name,
    settings: WarmupIterations(10)
  ) { suite in
    for batchSize in [32, 64, 128, 256, 512, 1024] {
      for backend in [Backend(.x10), Backend(.eager)] {
        suite.benchmark(
          "forward_b\(batchSize)_\(backend.value)",
          settings: backend, BatchSize(batchSize),
          function: makeForwardBenchmark(layer: layer, inputDimensions: inp, outputDimensions: outp)
        )

        suite.benchmark(
          "forward_and_gradient_b\(batchSize)_\(backend.value)",
          settings: backend, BatchSize(batchSize),
          function: makeGradientBenchmark(
            layer: layer, inputDimensions: inp, outputDimensions: outp))
      }
    }
  }
}
