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
    filter: ckpt.load(from: "Conv2d_0/weights"),
    bias: ckpt.load(from: "Conv2d_0/biases"),
    activation: relu6,
    strides: (2, 2),
    padding: .same
)

var conv_1 = DepthwiseSeparableConvBlock(
    depthWiseFilter: ckpt.load(from: "Conv2d_1_depthwise/depthwise_weights"),
    depthWiseBias: ckpt.load(from: "Conv2d_1_depthwise/biases"),
    pointWiseFilter: ckpt.load(from: "Conv2d_1_pointwise/weights"),
    pointWiseBias: ckpt.load(from: "Conv2d_1_pointwise/biases"),
    strides: (1, 1)
)

var x = conv_0(normalizedImagesTensor)
x = conv_1(x)


h(x)
