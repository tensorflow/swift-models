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

    init(imageDirectory: URL, imageSize: (Int32, Int32)) throws {
        let dirContents = try FileManager.default.contentsOfDirectory(at: imageDirectory,
            includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        var newImageData: [Float] = []
        var newLabels: [String] = []
        var newImageLabels: [Int32] = []
        var currentLabel: Int32 = 0

        for directoryURL in dirContents {
            newLabels.append(directoryURL.lastPathComponent)
            let subdirContents = try FileManager.default.contentsOfDirectory(at: directoryURL,
                includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
            for fileURL in subdirContents {
                if let imageFloats = loadCenterCroppedImageUsingTensorFlow(from: fileURL,
                    size: imageSize) {
                    newImageData.append(contentsOf: imageFloats)
                    newImageLabels.append(currentLabel)
                }
            }
            currentLabel += 1
        }

        self.classes = newLabels.count
        self.labels = newLabels
        self.imageData = Tensor<Float>(
            shape:[Int(newImageLabels.count), Int(imageSize.0), Int(imageSize.1), 3],
            scalars: newImageData)
        self.imageLabels = Tensor<Int32>(newImageLabels)

        self.combinedDataset =  Dataset(elements: Example(label: Tensor<Int32>(self.imageLabels),
            data: self.imageData))
    }
}

func loadCenterCroppedImageUsingTensorFlow(from fileURL: URL, size: (Int32, Int32)) -> [Float]? {
    let loadedFile = Raw.readFile(filename: StringTensor(fileURL.absoluteString))
    let loadedJpeg = Raw.decodeJpeg(contents: loadedFile, channels: 3, dctMethod: "")

    let maxX: Float = Float(loadedJpeg.shape[0])
    let maxY: Float = Float(loadedJpeg.shape[1])

    let xPrime: Float = (Float(maxX) - Float(size.0))/2.0
    let yPrime: Float = (Float(maxY) - Float(size.1))/2.0

    let xOne = xPrime/maxX
    let yOne = yPrime/maxY
    let xTwo = (xPrime + Float(size.0))/maxX
    let yTwo = (yPrime + Float(size.1))/maxY

    let boxesWrapped = Tensor<Float>(shape:[1, 4], scalars: [yOne, xOne, yTwo, xTwo])
    let boxIndicies = Tensor<Int32>([0])

    let centerCroppedImage = Raw.cropAndResize(image: Tensor<UInt8>([loadedJpeg]),
        boxes: boxesWrapped, boxInd: boxIndicies,
        cropSize:Tensor<Int32>([Int32(size.0),Int32(size.1)]))
    return centerCroppedImage.scalars
}
