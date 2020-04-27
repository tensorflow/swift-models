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

import TensorFlow

public protocol EasyLayers {
  typealias Conv2D = TensorFlow.Conv2D<Float>
  typealias AvgPool2D = TensorFlow.AvgPool2D<Float>
  typealias MaxPool2D = TensorFlow.MaxPool2D<Float>
  typealias Flatten = TensorFlow.Flatten<Float>
  typealias Dense = TensorFlow.Dense<Float>
  typealias TensorF = TensorFlow.Tensor<Float>
}


// Something closer to the actual design...
public protocol TypedModel: Layer {
  associatedtype Scalar: TensorFlowScalar
  
  typealias Tensor = TensorFlow.Tensor<Scalar>
  typealias TensorF = TensorFlow.Tensor<Float>
  typealias TTensor = TensorFlow.Tensor
}

public extension TypedModel where Scalar: TensorFlowFloatingPoint {
  typealias Conv2D = TensorFlow.Conv2D<Scalar>
  typealias AvgPool2D = TensorFlow.AvgPool2D<Scalar>
  typealias Flatten = TensorFlow.Flatten<Scalar>
  typealias Dense = TensorFlow.Dense<Scalar>
}

public protocol FloatModel: TypedModel where Scalar == Float {}
