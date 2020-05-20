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
    self.ckpt = try! CheckpointReader(
      checkpointLocation: URL(fileURLWithPath: config.checkpointPath), modelName: "Personlab"
    )
    self.backbone = MobileNetLikeBackbone(checkpoint: ckpt)
    self.personlabHeads = PersonlabHeads(checkpoint: ckpt)
  }

  public func callAsFunction(_ inputImage: Image) -> [Pose] {
    let startTime = Date()
    // Careful, `resized` adds a batch dimension automagically.
    let resizedImages = inputImage.resized(to: config.inputImageSize)
    let normalizedImagesTensorBGR = resizedImages.tensor * (2.0 / 255.0) - 1.0
    let normalizedImagesTensorRGB = _Raw.reverse(normalizedImagesTensorBGR, dims: [false, false, false, true])
    let preprocessingTime = Date()

    let convnetResults = personlabHeads(backbone(normalizedImagesTensorRGB))
    let convnetTime = Date()

    let poseDecoder = PoseDecoder(for: convnetResults, with: self.config)
    let poses = poseDecoder.decode()
    let decoderTime = Date()

    if self.config.printProfilingData {
      print("Preprocessing seconds (slow on first iteration):", preprocessingTime.timeIntervalSince(startTime))
      print("Convnet seconds:", convnetTime.timeIntervalSince(preprocessingTime))
      print("Decoder seconds:", decoderTime.timeIntervalSince(convnetTime))
    }

    return poses 
  }

}
