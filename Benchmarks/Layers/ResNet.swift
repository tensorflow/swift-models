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

import ImageClassificationModels

extension ResNet50: DefaultInit {}

extension ResNet56: DefaultInit {}

let ResNet50_224x224x3_1000 = makeLayerSuite(
  layer: ResNet50.self,
  inputDimensions: [224, 224, 3],
  outputDimensions: [1000])

let ResNet56_32x32x3_10 = makeLayerSuite(
  layer: ResNet56.self,
  inputDimensions: [32, 32, 3],
  outputDimensions: [10])

let ResNetSuites = [
  ResNet50_224x224x3_1000,
  ResNet56_32x32x3_10,
]
