import Batcher
import Foundation

public struct COCODataset: ObjectDetectionDataset {
    public typealias SourceDataSet = [ObjectDetectionExample]
    public let trainingExamples: SourceDataSet
    public let training: Batcher<SourceDataSet>
    public let testExamples: SourceDataSet
    public let test: Batcher<SourceDataSet>

    public init(includeMasks: Bool, batchSize: Int, numWorkers: Int) {
        self.trainingExamples =
            loadCOCOExamples(
                from: COCOVariant.loadTrain(),
                includeMasks: includeMasks,
                batchSize: batchSize,
                numWorkers: numWorkers)
        self.training =
            Batcher(
                on: trainingExamples,
                batchSize: batchSize,
                numWorkers: numWorkers,
                shuffle: true)
        self.testExamples =
            loadCOCOExamples(
                from: COCOVariant.loadVal(),
                includeMasks: includeMasks,
                batchSize: batchSize,
                numWorkers: numWorkers)
        self.test =
            Batcher(
                on: testExamples,
                batchSize: batchSize,
                numWorkers: numWorkers,
                shuffle: false)
    }
}

func loadCOCOExamples(from coco: COCO, includeMasks: Bool, batchSize: Int, numWorkers: Int)
    -> [ObjectDetectionExample]
{
    let images = coco.metadata["images"] as! [COCO.Image]
    let batchCount: Int = images.count / batchSize + 1
    let n = min(numWorkers, batchCount)
    let batches = Array(0..<batchCount)
    let examples: [[ObjectDetectionExample]] = batches.concurrentMap(nthreads: n) { batchIdx in
        var examples: [ObjectDetectionExample] = []
        for i in 0..<batchSize {
            let idx = batchSize * batchIdx + i
            if idx < images.count {
                let img = images[idx]
                let example = loadCOCOExample(coco: coco, image: img, includeMasks: includeMasks)
                examples.append(example)
            }
        }
        return examples
    }
    let result = Array(examples.joined())
    assert(result.count == images.count)
    return result
}

func loadCOCOExample(coco: COCO, image: COCO.Image, includeMasks: Bool) -> ObjectDetectionExample {
    let imgDir = coco.imagesDirectory
    let imgW = image["width"] as! Int
    let imgH = image["height"] as! Int
    let imgFileName = image["file_name"] as! String
    var imgUrl: URL? = nil
    if imgDir != nil {
        let imgPath = imgDir!.appendingPathComponent(imgFileName).path
        imgUrl = URL(string: imgPath)!
    }
    let imgId = image["id"] as! Int
    let img = LazyImage(width: imgW, height: imgH, url: imgUrl)
    let annotations: [COCO.Annotation]
    if let anns = coco.imageToAnnotations[imgId] {
        annotations = anns
    } else {
        annotations = []
    }
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
        let classInfo = coco.categories[classId]!
        let className = classInfo["name"] as! String
        let maskRLE: RLE?
        if includeMasks {
            maskRLE = coco.annotationToRLE(annotation)
        } else {
            maskRLE = nil
        }
        let object = LabeledObject(
            xMin: xMin, xMax: xMax,
            yMin: yMin, yMax: yMax,
            className: className, classId: classId,
            isCrowd: isCrowd, area: area, maskRLE: maskRLE)
        objects.append(object)
    }
    return ObjectDetectionExample(image: img, objects: objects)
}
