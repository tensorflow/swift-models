import Foundation
import TensorFlow
import ModelSupport
import ImageClassificationModels

// Setup
let config = Config()
Context.local.learningPhase = .inference
let checkpointFile = URL(fileURLWithPath: "/Users/joaqo/code/personlab/checkpoints/Personlab")
var ckpt = try! CheckpointReader(checkpointLocation: checkpointFile, modelName: "Personlab")

// Pre-processing
var image = Image(jpeg: URL(fileURLWithPath: config.testImagePath))
image = image.resized(to: config.inputSize)  // Adds a batch dimension automagically
var normalizedImagesTensor = image.tensor * (2.0 / 255.0) - 1.0

// Define and run convnet
let backbone = MobileNetLikeBackbone(checkpoint: ckpt)
let personlabHeads = PersonlabHeads(checkpoint: ckpt)

var results = personlabHeads(backbone(normalizedImagesTensor))


var priorityQueueTest = Heap<Int>(elements:[10, 22, 3, 1, 5, 7, 7, 3] , priorityFunction: >)
print(priorityQueueTest.dequeue()!)
print(priorityQueueTest.dequeue()!)
print(priorityQueueTest.dequeue()!)
print(priorityQueueTest.dequeue()!)
print(priorityQueueTest.dequeue()!)
