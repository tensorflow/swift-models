import Foundation
import TensorFlow
import ModelSupport
import SwiftCV
import ArgumentParser


struct Inference: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "personlab",
    abstract: """
    Runs human pose estimation on a local image file or on a local webcam.
    """
  )

  @Argument(help: "Path to checkpoint directory")
  var checkpointPath: String

  @Option(name: .shortAndLong, help: "Path to local image to run pose estimation on")
  var imagePath: String?

  @Flag(name: .shortAndLong, help: "Run local webcam demo")
  var webcamDemo: Bool

  @Flag(name: .shortAndLong, help: "Print profiling data")
  var profiling: Bool

  func run() {
    Context.local.learningPhase = .inference
    let config = Config(checkpointPath: checkpointPath, printProfilingData: profiling)
    let model = PersonLab(config)

    if let imagePath = imagePath {
      let fileManager = FileManager()
      if !fileManager.fileExists(atPath: imagePath) {
        print("No image found at path: \(imagePath)")
        return
      }
      let swiftcvImage = imread(imagePath)
      let image = Image(tensor: Tensor<UInt8>(cvMat: swiftcvImage)!)
      let poses = model(image)

      for pose in poses {
        draw(pose, on: swiftcvImage, color: config.color, lineWidth: config.lineWidth)
      }
      ImShow(image: swiftcvImage)
      WaitKey(delay: 0)
    }

    if webcamDemo {
      let videoCaptureDevice = VideoCapture(0)
      videoCaptureDevice.set(VideoCaptureProperties.CAP_PROP_BUFFERSIZE, 1)  // Reduces latency
      
      let frame = Mat()
      while true {
        videoCaptureDevice.read(into: frame)
        let image = Image(tensor: Tensor<UInt8>(cvMat: frame)!)
        let poses = model(image)

        for pose in poses {
          draw(pose, on: frame, color: config.color, lineWidth: config.lineWidth)
        }
        ImShow(image: frame)
        WaitKey(delay: 1)
      }
    }
  }
}

Inference.main()
