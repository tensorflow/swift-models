import Batcher
import Foundation

public struct COCODataset {
    public typealias SourceDataSet = [ObjectDetectionExample]
    public let batchSize: Int
    public let trainingExamples: SourceDataSet
    public let testExamples: SourceDataSet

    public lazy var training: Batcher<SourceDataSet> = {
        return Batcher(
            on: trainingExamples,
            batchSize: batchSize, numWorkers: 1,
            shuffle: true)
    }()

    public lazy var test: Batcher<SourceDataSet> = {
        return Batcher(
            on: testExamples,
            batchSize: batchSize,
            numWorkers: 1,
            shuffle: true)
    }()

    public init(batchSize: Int) {
        self.init(batchSize: batchSize, includeMasks: false)
    }

    public init(batchSize: Int, includeMasks: Bool = false) {
        self.batchSize = batchSize
        self.trainingExamples =
            loadCOCOExamples(
                from: COCOVariant.loadInstancesTrain2017(),
                includeMasks: includeMasks)
        self.testExamples =
            loadCOCOExamples(
                from: COCOVariant.loadInstancesVal2017(),
                includeMasks: includeMasks)
    }
}

func loadCOCOExamples(from coco: COCO, includeMasks: Bool = false) -> [ObjectDetectionExample] {
    var examples: [ObjectDetectionExample] = []
    examples.reserveCapacity(coco.images.count)
    for (_, image) in coco.images {
        let imgW = image["width"] as! Int
        let imgH = image["height"] as! Int
        let imgUrl = URL(string: image["file_name"] as! String)!
        let imgId = image["id"] as! Int
        let img = LazyImage(width: imgW, height: imgH, url: imgUrl)
        let annotations = coco.imageToAnnotations[imgId]!
        var objects: [LabeledObject] = []
        objects.reserveCapacity(annotations.count)
        for annotation in annotations {
            let bb = annotation["bbox"] as! [Double]
            let bbX = bb[0]
            let bbY = bb[1]
            let bbW = bb[2]
            let bbH = bb[3]
            let xMin = Float(bbX) / Float(imgW)
            let xMax = Float(bbX + bbW) / Float(imgW)
            let yMin = Float(bbY) / Float(imgH)
            let yMax = Float(bbY + bbH) / Float(imgH)
            let isCrowd: Int?
            if let iscrowd = annotation["iscrowd"] {
                isCrowd = iscrowd as? Int
            } else {
                isCrowd = nil
            }
            let area = Float(annotation["area"] as! Double)
            let classId = annotation["category_id"] as! Int
            let className = coco.annotations[classId]!["name"] as! String
            let mask: Mask?
            if includeMasks {
                mask = coco.annotationToMask(annotation)
            } else {
                mask = nil
            }
            let object = LabeledObject(
                xMin: xMin, xMax: xMax,
                yMin: yMin, yMax: yMax,
                className: className, classId: classId,
                isCrowd: isCrowd, area: area, mask: mask)
            objects.append(object)
        }
        let example = ObjectDetectionExample(image: img, objects: objects)
        examples.append(example)
    }
    return examples
}
