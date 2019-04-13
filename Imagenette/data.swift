import TensorFlow
import Foundation

struct Example: TensorGroup {
    var label: Tensor<Int32>
    var data: Tensor<Float>
}

struct ImageDataset {
    let classes: Int
    let labels: [String]
    let imageData: Tensor<Float>
    let imageLabels: Tensor<Int32>
    let combinedDataset: Dataset<Example>

    init(imageDirectory: URL, imageSize: (Int, Int)) throws {
        let dirContents = try FileManager.default.contentsOfDirectory(at:imageDirectory, includingPropertiesForKeys: [.isDirectoryKey], options:[.skipsHiddenFiles])
        var newImageData: [Float] = []
        var newLabels: [String] = []
        var newImageLabels: [Int32] = []
        var currentLabel: Int32 = 0

        for directoryURL in dirContents {
            newLabels.append(directoryURL.lastPathComponent)
            let subdirContents = try FileManager.default.contentsOfDirectory(at:directoryURL, includingPropertiesForKeys: [.isDirectoryKey], options:[.skipsHiddenFiles])
            for fileURL in subdirContents {
                if let imageFloats = loadImageUsingTensorFlow(from: fileURL, size: imageSize) {
                    newImageData.append(contentsOf: imageFloats)
                    newImageLabels.append(currentLabel)
                }
            }
            currentLabel += 1
        }

        self.classes = newLabels.count
        self.labels = newLabels
        self.imageData = Tensor<Float>(shape:[Int32(newImageLabels.count), Int32(imageSize.0), Int32(imageSize.1), 3], scalars: newImageData)
        self.imageLabels = Tensor<Int32>(newImageLabels)

        self.combinedDataset =  Dataset(elements: Example(label: Tensor<Int32>(self.imageLabels), data: self.imageData))
    }
}

func loadImageUsingTensorFlow(from fileURL: URL, size: (Int, Int)) -> [Float]? {
    let loadedFile = Raw.readFile(filename: StringTensor(fileURL.absoluteString))
    let loadedJpeg = Raw.decodeJpeg(contents: loadedFile, channels: 3, dctMethod: "")
    let resizedImage = Raw.resizeBilinear(images: Tensor<UInt8>([loadedJpeg]), size: Tensor<Int32>([Int32(size.0), Int32(size.1)]))
    let reversedChannelImage = Raw.reverse(resizedImage, dims: Tensor<Bool>([false, false, false, true]))
    return reversedChannelImage.scalars
}
