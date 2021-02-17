import ModelSupport
import TensorFlow
import TrainingLoop

/// Returns a callback that saves images that `imageWidth` in width and `imageHeight` in height;
/// The callback saves an input and an output image once per epoch in validation phrase;
/// It's ensured that each epoch will save different images as long as 
/// count of epochs is less or equal than count of images; `batchSize` is used for hashing the 
/// image used for current state of `loop`.
public func imageSaver<L: TrainingLoopProtocol>(
  batchSize: Int, imageWidth: Int, imageHeight: Int
) -> TrainingLoopCallback<L> {
  return { (loop, event) throws -> Void in
    if event != .inferencePredictionEnd { return }

    guard let batchIndex = loop.batchIndex,
      let batchCount = loop.batchCount,
      let epochIndex = loop.epochIndex,
      let epochCount = loop.epochCount,
      let input = loop.lastStepInput,
      let output = loop.lastStepOutput
    else {
      return
    }

    let imageCount = batchCount * batchSize
    let selectedImageGlobalIndex = epochIndex * (imageCount / epochCount)
    let selectedBatchIndex = selectedImageGlobalIndex / batchSize

    if batchIndex != selectedBatchIndex { return }

    let outputFolder = "./output/"
    let selectedImageBatchLocalIndex = selectedImageGlobalIndex % batchSize
    let inputExample = (input as! Tensor<Float>)[
      selectedImageBatchLocalIndex..<selectedImageBatchLocalIndex+1]
      .normalizedToGrayscale().reshaped(to: [imageWidth, imageHeight, 1])
    try inputExample.saveImage(
      directory: outputFolder, name: "epoch-\(epochIndex + 1)-of-\(epochCount)-input", format: .png)
    let outputExample = (output as! Tensor<Float>)[
      selectedImageBatchLocalIndex..<selectedImageBatchLocalIndex+1]
      .normalizedToGrayscale().reshaped(to: [imageWidth, imageHeight, 1])
    try outputExample.saveImage(
      directory: outputFolder, name: "epoch-\(epochIndex + 1)-of-\(epochCount)-output", format: .png
    )
  }
}
