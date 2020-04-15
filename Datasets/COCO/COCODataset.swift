import Batcher

public struct COCODataset {
    public typealias SourceDataSet = [ObjectDetectionExample]
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>

    public init(batchSize: Int) {
        self.training = Batcher(
            on: loadExamples(COCOVariant.loadInstancesTrain2017()),
            batchSize: batchSize, numWorkers: 1,
            shuffle: true)
        self.test = Batcher(
            on: loadExamples(COCOVariant.loadInstancesVal2017()),
            batchSize: batchSize,
            numWorkers: 1,
            shuffle: true)
    }
}

func loadExamples(_ coco: COCO) -> [ObjectDetectionExample] {
    fatalError("todo")
}
