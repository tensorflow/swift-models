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

let ResNet50Suite = makeLayerSuite(
  layer: ResNet50.self,
  inputDimensions: ResNet50.preferredInputDimensions,
  outputDimensions: [ResNet50.outputLabels])

let ResNet56Suite = makeLayerSuite(
  layer: ResNet56.self,
  inputDimensions: ResNet56.preferredInputDimensions,
  outputDimensions: [ResNet56.outputLabels])
