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

let ResNetSuites = [
  //
  // Cifar input dimensions. 
  //
  makeLayerSuite(
    name: "ResNet18",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) {
    ResNet(classCount: 10, depth: .resNet18, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name: "ResNet34",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) {
    ResNet(classCount: 10, depth: .resNet34, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name: "ResNet50",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) {
    ResNet(classCount: 10, depth: .resNet50, downsamplingInFirstStage: true, useLaterStride: false)
  },
  makeLayerSuite(
    name: "ResNet101",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) {
    ResNet(classCount: 10, depth: .resNet101, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name: "ResNet152",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) {
    ResNet(classCount: 10, depth: .resNet152, downsamplingInFirstStage: true)
  },
  //
  // ImageNet dimensions.
  //
  makeLayerSuite(
    name: "ResNet18",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNet(classCount: 1000, depth: .resNet18)
  },
  makeLayerSuite(
    name: "ResNet34",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNet(classCount: 1000, depth: .resNet34)
  },
  makeLayerSuite(
    name: "ResNet50",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNet(classCount: 1000, depth: .resNet50, useLaterStride: false)
  },
  makeLayerSuite(
    name: "ResNet101",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNet(classCount: 1000, depth: .resNet101)
  },
  makeLayerSuite(
    name: "ResNet152",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNet(classCount: 1000, depth: .resNet152)
  },
]
