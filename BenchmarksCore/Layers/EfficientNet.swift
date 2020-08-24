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

let EfficientNetSuites = [
  makeLayerSuite(
    name: "EfficientNetB0",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB0, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB1",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB1, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB2, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB3",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB3, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB4",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB4, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB5",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB5, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB6",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB6, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB7",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB7, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetB8",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetB8, classCount: 1000)
  },
  makeLayerSuite(
    name: "EfficientNetL2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    EfficientNet(kind: .efficientnetL2, classCount: 1000)
  },
]
