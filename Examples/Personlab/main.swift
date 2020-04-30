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
image = image.resized(to: config.inputImageSize)  // Adds a batch dimension automagically
var normalizedImagesTensor = image.tensor * (2.0 / 255.0) - 1.0

// Define network and load pre trained weights
let backbone = MobileNetLikeBackbone(checkpoint: ckpt)
let personlabHeads = PersonlabHeads(checkpoint: ckpt)

// Run pose estimator
var convnetResults = personlabHeads(backbone(normalizedImagesTensor))
let poseDecoder = PoseDecoder(for: convnetResults, with: config)
var poses = poseDecoder.decode()

// Results
for pose in poses {
  print(pose)
}
print("Number of poses", poses.count)
