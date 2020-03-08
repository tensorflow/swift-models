import TensorFlow
import Files
import Foundation

public class Facades {
    struct ImagePair: TensorGroup {
        var sourceImages: Tensor<Float>
        var targetImages: Tensor<Float>
    }

    let dataset: Dataset<ImagePair>
    let count: Int

    public init(folder: Folder) throws {
        let imageFiles = folder.files(extensions: ["jpg"])

        var sourceData: [Float] = []
        var targetData: [Float] = []

        var pairs = 0

        for imageFile in imageFiles {
            let cumulativeImageTensor = Image(jpeg: imageFile.url).tensor

            let sourceImage = cumulativeImageTensor.slice(lowerBounds: [0, 256, 0], sizes: [256, 256, 3])
            let targetImage = cumulativeImageTensor.slice(lowerBounds: [0, 0, 0], sizes: [256, 256, 3])

            sourceData.append(contentsOf: sourceImage.scalars)
            targetData.append(contentsOf: targetImage.scalars)

            pairs += 1
        }

        let source = Tensor<Float>(shape: [pairs, 256, 256, 3], scalars: sourceData) / 127.5 - 1.0
        let target = Tensor<Float>(shape: [pairs, 256, 256, 3], scalars: targetData) / 127.5 - 1.0

        self.dataset = .init(elements: ImagePair(sourceImages: source, targetImages: target))
        self.count = pairs
    }
}
