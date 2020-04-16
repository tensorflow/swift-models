import Foundation
import TensorFlow
import ModelSupport
import ImageClassificationModels

// Setup
let config = Config()
Context.local.learningPhase = .inference
let checkpointFile = URL(fileURLWithPath: "/Users/joaqo/code/personlab/checkpoints/Personlab")
var ckpt = try! CheckpointReader(checkpointLocation: checkpointFile, modelName: "Personlab")

// Preprocessing
var image = Image(jpeg: URL(fileURLWithPath: config.testImagePath))
image = image.resized(to: config.inputSize)  // Adds a batch dimension automagically
var normalizedImagesTensor = image.tensor * (2.0 / 255.0) - 1.0

// Define backbone
var conv_0 = Conv2D<Float>(
    filterShape: (3, 3, 3, 24),
    strides: (2, 2),
    padding: .same
)
conv_0.bias = Tensor(ckpt.loadTensor(named: "MobilenetV1/Conv2d_0/biases"))
conv_0.filter = Tensor(ckpt.loadTensor(named: "MobilenetV1/Conv2d_0/weights"))

var convOutput = relu6( conv_0(normalizedImagesTensor))
h(convOutput)
