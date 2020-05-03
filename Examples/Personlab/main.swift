import Foundation
import TensorFlow
import ModelSupport
import SwiftCV

// Setup
let config = Config()
Context.local.learningPhase = .inference
let checkpointFile = URL(fileURLWithPath: config.checkPointPath)
var ckpt = try! CheckpointReader(checkpointLocation: checkpointFile, modelName: "Personlab")

// // Pre-processing
// var image = Image(jpeg: URL(fileURLWithPath: config.testImagePath))
// image = image.resized(to: config.inputImageSize)  // Adds a batch dimension automagically
// var normalizedImagesTensor = image.tensor * (2.0 / 255.0) - 1.0

var image = Image(tensor: Tensor<UInt8>(cvMat: imread(config.testImagePath))!)
image = image.resized(to: config.inputImageSize)  // Adds a batch dimension automagically
var normalizedImagesTensor = image.tensor * (2.0 / 255.0) - 1.0
normalizedImagesTensor = _Raw.reverse(normalizedImagesTensor, dims: [false, false, false, true])

// Define network and load pre trained weights
let backbone = MobileNetLikeBackbone(checkpoint: ckpt)
let personlabHeads = PersonlabHeads(checkpoint: ckpt)

// Run pose estimator
let startTime = Date()
var convnetResults = personlabHeads(backbone(normalizedImagesTensor))
let convnetTime = Date()
let poseDecoder = PoseDecoder(for: convnetResults, with: config)
var poses = poseDecoder.decode()

for pose in poses {
  print(pose)
}

print("Number of poses", poses.count)
print("Convnet seconds", convnetTime.timeIntervalSince(startTime))
print("Decoder seconds", Date().timeIntervalSince(convnetTime))

// let cap = VideoCapture(0)
// // Optional, reduces latency a bit
// cap.set(VideoCaptureProperties.CAP_PROP_BUFFERSIZE, 1)
// 
// let frame = Mat()
// let point1 = Point(x: 1, y: 10)
// let point2 = Point(x: 302, y: 402)
// let point3 = Point(x: 200, y: 402)
// let color = Scalar(val1: 0, val2: 255, val3: 255, val4: 1)
// 
// while true {
//   // Get tensor
//   cap.read(into: frame)
//   var image = Image(tensor: Tensor<UInt8>(cvMat: frame)!)
//   image = image.resized(to: config.inputImageSize)  // Adds a batch dimension automagically
//   let normalizedImagesTensor = image.tensor * (2.0 / 255.0) - 1.0
// 
//   // Run pose estimator
//   let startTime = Date()
//   let convnetResults = personlabHeads(backbone(normalizedImagesTensor))
//   let convnetTime = Date()
//   let poseDecoder = PoseDecoder(for: convnetResults, with: config)
//   let poses = poseDecoder.decode()
//   line(img: frame, pt1: point1, pt2: point2, color: color, thickness: 3)
//   ImShow(image: frame)
//   WaitKey(delay: 1)
// 
//   // Results
//   print("Number of poses", poses.count)
//   print("Convnet seconds", convnetTime.timeIntervalSince(startTime))
//   print("Decoder seconds", Date().timeIntervalSince(convnetTime))
// }
