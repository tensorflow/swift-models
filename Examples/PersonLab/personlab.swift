import TensorFlow
import ModelSupport
import Foundation

public struct PersonLab {
  let config: Config
  let ckpt: CheckpointReader
  let backbone: MobileNetLikeBackbone
  let personlabHeads: PersonlabHeads

  public init(_ config: Config) {
    self.config = config
    do {
      self.ckpt = try CheckpointReader(
        checkpointLocation: URL(fileURLWithPath: config.checkpointPath), modelName: "Personlab"
      )
    } catch {
      print("Error loading checkpoint file: \(config.checkpointPath)")
      print(error)
      exit(0)
    }
    self.backbone = MobileNetLikeBackbone(checkpoint: ckpt)
    self.personlabHeads = PersonlabHeads(checkpoint: ckpt)
  }

  public func callAsFunction(_ inputImage: Image) -> [Pose] {
    // Careful, `resized` adds a batch dimension automagically.
    let resizedImages = inputImage.resized(to: config.inputImageSize)
    let normalizedImagesTensorBGR = resizedImages.tensor * (2.0 / 255.0) - 1.0
    let normalizedImagesTensorRGB = _Raw.reverse(normalizedImagesTensorBGR, dims: [false, false, false, true])

    let startTime = Date()
    let convnetResults = personlabHeads(backbone(normalizedImagesTensorRGB))
    let convnetTime = Date()

    let poseDecoder = PoseDecoder(for: convnetResults, with: self.config)
    let poses = poseDecoder.decode()
    let decoderTime = Date()

    if self.config.printProfilingData {
      print(
        String(format: "Backbone: %.2f ms", convnetTime.timeIntervalSince(startTime) * 1000),
        "|",
        String(format: "Decoder: %.2f ms", decoderTime.timeIntervalSince(convnetTime) * 1000)
      )
    }

    return poses 
  }

}
