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

    init(imageDirectory: URL, imageSize: (Int, Int), trainingData: Bool) throws {
        let dirContents = try FileManager.default.contentsOfDirectory(at: imageDirectory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        var newImageData: [Float] = []
        var newLabels: [String] = []
        var newImageLabels: [Int32] = []
        var currentLabel: Int32 = 0

        for directoryURL in dirContents {
            newLabels.append(directoryURL.lastPathComponent)
            let subdirContents = try FileManager.default.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
            for fileURL in subdirContents {
                if (trainingData) {
                    for _ in 1...4 {
                        if let imageFloats = loadRandomlyCroppedImageUsingTensorFlow(from: fileURL, size: imageSize) {
                            newImageData.append(contentsOf: imageFloats)
                            newImageLabels.append(currentLabel)
                        }
                    }
                } else {
                    if let imageFloats = loadCenterCroppedImageUsingTensorFlow(from: fileURL, size: imageSize) {
                        newImageData.append(contentsOf: imageFloats)
                        newImageLabels.append(currentLabel)
                    }
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

func loadRandomlyCroppedImageUsingTensorFlow(from fileURL: URL, size: (Int, Int)) -> [Float]? {
    let loadedFile = Raw.readFile(filename: StringTensor(fileURL.absoluteString))
    let loadedJpeg = Raw.decodeJpeg(contents: loadedFile, channels: 3, dctMethod: "")

    let maxX: Int32 = Int32(loadedJpeg.shape[0]) - Int32(size.0)
    let maxY: Int32 = Int32(loadedJpeg.shape[1]) - Int32(size.1)
    let xOffset = Float(Int32.random(in: 0..<maxX))/Float(size.0)
    let yOffset = Float(Int32.random(in: 0..<maxY))/Float(size.1)

    let boxesWrapped = Tensor<Float>(shape:[1, 4], scalars: [yOffset, xOffset, 1-yOffset, 1-xOffset])
    let boxIndicies = Tensor<Int32>([0])

    let randomlyCroppedImage = Raw.cropAndResize(image: Tensor<UInt8>([loadedJpeg]), boxes: boxesWrapped, boxInd: boxIndicies, cropSize:Tensor<Int32>([Int32(size.0),Int32(size.1)]))
    return randomlyCroppedImage.scalars
}


func loadCenterCroppedImageUsingTensorFlow(from fileURL: URL, size: (Int, Int)) -> [Float]? {
    let loadedFile = Raw.readFile(filename: StringTensor(fileURL.absoluteString))
    let loadedJpeg = Raw.decodeJpeg(contents: loadedFile, channels: 3, dctMethod: "")

    let maxX: Int32 = Int32(loadedJpeg.shape[0]) - Int32(size.0)
    let maxY: Int32 = Int32(loadedJpeg.shape[1]) - Int32(size.1)
    let xOffset = Float(maxX/2)/Float(size.0)
    let yOffset = Float(maxY/2)/Float(size.1)

    let boxesWrapped = Tensor<Float>(shape:[1, 4], scalars: [yOffset, xOffset, 1-yOffset, 1-xOffset])
    let boxIndicies = Tensor<Int32>([0])

    let randomlyCroppedImage = Raw.cropAndResize(image: Tensor<UInt8>([loadedJpeg]), boxes: boxesWrapped, boxInd: boxIndicies, cropSize:Tensor<Int32>([Int32(size.0),Int32(size.1)]))
    return randomlyCroppedImage.scalars
}
