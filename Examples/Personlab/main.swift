import Foundation
import TensorFlow
import ModelSupport
import SwiftCV
import ArgumentParser

// TODO: Rethink this global config
let config = Config()

struct InferenceCommand: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "Personlab Human Pose Estimator",
    abstract: """
    Runs human pose estimation on a local image file or on a local webcam.
    """
  )

  @Argument(help: "Path to checkpoint directory")
  var checkpointPath: String

  @Option(name: .shortAndLong, help: "Path to local image to run pose estimation on")
  var imagePath: String?

  @Flag(help: "Webcam demo")
  var webcamDemo: Bool

  func run() {
    Context.local.learningPhase = .inference
    let checkpointFile = URL(fileURLWithPath: checkpointPath)
    let ckpt = try! CheckpointReader(checkpointLocation: checkpointFile, modelName: "Personlab")

    // Define network and load pre trained weights
    let backbone = MobileNetLikeBackbone(checkpoint: ckpt)
    let personlabHeads = PersonlabHeads(checkpoint: ckpt)

    if let imagePath = imagePath {
      // Get preprocessed tensor
      let frame = imread(imagePath)
      var image = Image(tensor: Tensor<UInt8>(cvMat: frame)!)
      image = image.resized(to: config.inputImageSize)  // Adds a batch dimension automagically
      let normalizedImagesTensorBGR = image.tensor * (2.0 / 255.0) - 1.0
      let normalizedImagesTensorRGB = _Raw.reverse(normalizedImagesTensorBGR, dims: [false, false, false, true])

      // Run pose estimator
      let startTime = Date()
      let convnetResults = personlabHeads(backbone(normalizedImagesTensorRGB))
      let convnetTime = Date()
      let poseDecoder = PoseDecoder(for: convnetResults, with: config)
      let poses = poseDecoder.decode()
      print("Convnet seconds", convnetTime.timeIntervalSince(startTime))
      print("Decoder seconds", Date().timeIntervalSince(convnetTime))

      // Draw Results
      for pose in poses {
        draw(pose, on: frame)
      }
      ImShow(image: frame)
      WaitKey(delay: 5000)
    }

    if webcamDemo {
      let videoCaptureDevice = VideoCapture(0)
      // Optional, reduces latency a bit
      videoCaptureDevice.set(VideoCaptureProperties.CAP_PROP_BUFFERSIZE, 1)
      
      let frame = Mat()
      while true {
        // Get preprocessed tensor
        videoCaptureDevice.read(into: frame)
        var image = Image(tensor: Tensor<UInt8>(cvMat: frame)!)
        image = image.resized(to: config.inputImageSize)  // Adds a batch dimension automagically
        let normalizedImagesTensorBGR = image.tensor * (2.0 / 255.0) - 1.0
        let normalizedImagesTensorRGB = _Raw.reverse(normalizedImagesTensorBGR, dims: [false, false, false, true])
      
        // Run pose estimator
        let convnetResults = personlabHeads(backbone(normalizedImagesTensorRGB))
        let poseDecoder = PoseDecoder(for: convnetResults, with: config)
        let poses = poseDecoder.decode()

        // Draw Results
        for pose in poses {
          draw(pose, on: frame)
        }
        ImShow(image: frame)
        WaitKey(delay: 1)
      }
    }
  }
}

InferenceCommand.main()
