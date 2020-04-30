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

// Define and run convnet
let backbone = MobileNetLikeBackbone(checkpoint: ckpt)
let personlabHeads = PersonlabHeads(checkpoint: ckpt)

let start = CFAbsoluteTimeGetCurrent()
var convnetResults = personlabHeads(backbone(normalizedImagesTensor))
print("Backbone", CFAbsoluteTimeGetCurrent() - start)
let startt = CFAbsoluteTimeGetCurrent()
let poseDecoder = PoseDecoder(for: convnetResults, with: config)
var poses = poseDecoder.decode()
print("Decoder", CFAbsoluteTimeGetCurrent() - startt)
for pose in poses {
  print(pose)
}
print("Number of poses", poses.count)
