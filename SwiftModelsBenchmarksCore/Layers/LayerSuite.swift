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
  layer makeLayer: @escaping () -> CustomLayer,
  inputDimensions: [Int],
  outputDimensions: [Int]
) -> ((inout BenchmarkState) throws -> Void)
where
  CustomLayer: Layer,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var layer = makeLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize: batchSize,
      dimensions: inputDimensions,
      device: device)

    var sink = makeRandomTensor(
      batchSize: batchSize, dimensions: outputDimensions, device: device)

    while true {
      do {
        try state.measure {
          let result = layer(input)
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
    // about the sink being never used.
    fatalError("unreachable \(sink)")
  }
}

func makeGradientBenchmark<CustomLayer>(
  layer makeLayer: @escaping () -> CustomLayer,
  inputDimensions: [Int],
  outputDimensions: [Int]
) -> ((inout BenchmarkState) throws -> Void)
where
  CustomLayer: Layer,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var layer = makeLayer()
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
    // about the sink being never used.
    fatalError("unrechable \(sink)")
  }
}

func makeLayerSuite<CustomLayer>(
  name: String,
  inputDimensions inp: [Int],
  outputDimensions outp: [Int],
  batchSizes: [Int] = [4],
  backends: [Backend.Value] = [.eager, .x10],
  layer: @escaping () -> CustomLayer
) -> BenchmarkSuite
where
  CustomLayer: Layer,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  let inputString = inp.map { String($0) }.joined(separator: "x")
  let outputString = outp.map { String($0) }.joined(separator: "x")

  return BenchmarkSuite(
    name: "\(name)_\(inputString)_\(outputString)",
    settings: WarmupIterations(10)
  ) { suite in
    for batchSize in batchSizes {
      for backend in backends {
        suite.benchmark(
          "forward_b\(batchSize)_\(backend)",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
            layer: layer, inputDimensions: inp, outputDimensions: outp))

        suite.benchmark(
          "forward_and_gradient_b\(batchSize)_\(backend)",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeGradientBenchmark(
            layer: layer, inputDimensions: inp, outputDimensions: outp))
      }
    }
  }
}
