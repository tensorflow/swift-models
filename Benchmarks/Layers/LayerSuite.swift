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

func makeSyntheticBatch<Model>(
  model: Model.Type,
  batchSize: Int,
  device: Device
) -> (Tensor<Float>, Tensor<Int32>)
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float
{
  let dataset = SyntheticImageDataset<SystemRandomNumberGenerator>(
    batchSize: batchSize,
    labels: model.outputLabels,
    dimensions: model.preferredInputDimensions,
    entropy: SystemRandomNumberGenerator(),
    device: device)

  for epochBatches in dataset.training {
    for batch in epochBatches {
      return (batch.data, batch.label)
    }
  }

  fatalError("unreachable")
}

func forwardBenchmark<Model>(
  model modelType: Model.Type
) -> ((inout BenchmarkState) throws -> Void)
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var model = Model()
    model.move(to: device)
    let (images, _) = makeSyntheticBatch(model: modelType, batchSize: batchSize, device: device)

    var sink = TensorShape([])

    while true {
      do {
        try state.measure {
          let result = model(images)
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

func gradientBenchmark<Model>(
  model modelType: Model.Type
) -> ((inout BenchmarkState) throws -> Void)
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var model = Model()
    model.move(to: device)
    let (images, labels) = makeSyntheticBatch(model: modelType, batchSize: batchSize, device: device)

    var sink: Model.TangentVector = Model.TangentVector.zero
    sink.move(to: device)

    while true {
      do {
        try state.measure {
          let result = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)
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

func updateBenchmark<Model>(
  model modelType: Model.Type
) -> ((inout BenchmarkState) throws -> Void)
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var model = Model()
    model.move(to: device)
    var optimizer = SGD(for: model, learningRate: 0.1)
    optimizer = SGD(copying: optimizer, to: device)
    let (images, labels) = makeSyntheticBatch(model: modelType, batchSize: batchSize, device: device)

    while true {
      do {
        try state.measure {
          let 𝛁model = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)
          }
          optimizer.update(&model, along: 𝛁model)
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
  }
}

func LayerSuite<Model>(
  model modelType: Model.Type
) -> BenchmarkSuite
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float
{
  let name: String = String(String(reflecting: modelType).split(separator: ".").last!)

  return BenchmarkSuite(
    name: name,
    settings: WarmupIterations(10)
  ) { suite in
    for batchSize in [32, 64, 128, 256, 512, 1024] {
      for backend in [Backend(.x10), Backend(.eager)] {
        suite.benchmark(
          "forward_b\(batchSize)_\(backend.value)",
          settings: backend, BatchSize(batchSize),
          function: forwardBenchmark(model: modelType))

        suite.benchmark(
          "gradient_b\(batchSize)_\(backend.value)",
          settings: backend, BatchSize(batchSize),
          function: gradientBenchmark(model: modelType))

        suite.benchmark(
          "update_b\(batchSize)_\(backend.value)",
          settings: backend, BatchSize(batchSize),
          function: updateBenchmark(model: modelType))
      }
    }
  }
}

let LeNetSuite = LayerSuite(model: LeNet.self)
let ResNet50Suite = LayerSuite(model: ResNet50.self)
let ResNet56Suite = LayerSuite(model: ResNet56.self)
