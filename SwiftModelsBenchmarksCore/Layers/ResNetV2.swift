import ImageClassificationModels

let ResNetV2Suites = [
  makeLayerSuite(
    name: "ResNet18v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNetV2(classCount: 1000, depth: .resNet18)
  },
  makeLayerSuite(
    name: "ResNet34v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNetV2(classCount: 1000, depth: .resNet34)
  },
  makeLayerSuite(
    name: "ResNet50v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNetV2(classCount: 1000, depth: .resNet50)
  },
  makeLayerSuite(
    name: "ResNet101v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNetV2(classCount: 1000, depth: .resNet101)
  },
  makeLayerSuite(
    name: "ResNet152v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) {
    ResNetV2(classCount: 1000, depth: .resNet152)
  },
]
