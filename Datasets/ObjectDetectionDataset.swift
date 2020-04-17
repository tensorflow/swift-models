import Batcher
import Foundation
import ModelSupport
import TensorFlow

public struct LazyImage {
    public let width: Int
    public let height: Int
    public let url: URL

    public init(width w: Int, height h: Int, url u: URL) {
        self.width = w
        self.height = h
        self.url = u
    }

    public func tensor() -> Tensor<Float> {
        return Image(jpeg: url).tensor
    }
}

public struct LabeledObject {
    public let xMin: Float
    public let xMax: Float
    public let yMin: Float
    public let yMax: Float
    public let className: String
    public let classId: Int
    public let isCrowd: Int?
    public let area: Float
    public let maskRLE: RLE?

    public init(
        xMin x0: Float, xMax x1: Float,
        yMin y0: Float, yMax y1: Float,
        className: String, classId: Int,
        isCrowd: Int?, area: Float, maskRLE: RLE?
    ) {
        self.xMin = x0
        self.xMax = x1
        self.yMin = y0
        self.yMax = y1
        self.className = className
        self.classId = classId
        self.isCrowd = isCrowd
        self.area = area
        self.maskRLE = maskRLE
    }
}

public struct ObjectDetectionExample: Collatable, KeyPathIterable {
    public let image: LazyImage
    public let objects: [LabeledObject]

    public init(image: LazyImage, objects: [LabeledObject]) {
        self.image = image
        self.objects = objects
    }
}

public protocol ObjectDetectionDataset {
    associatedtype SourceDataSet: Collection
    where SourceDataSet.Element == ObjectDetectionExample, SourceDataSet.Index == Int
    init(includeMasks: Bool, batchSize: Int, numWorkers: Int)
    var training: Batcher<SourceDataSet> { get }
    var test: Batcher<SourceDataSet> { get }
}
