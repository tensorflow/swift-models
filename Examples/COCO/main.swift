import Datasets
import Foundation
import SwiftProtobuf

let dataset = COCODataset(includeMasks: true, batchSize: 128, numWorkers: 72)
print(dataset.trainingExamples.count)
print(dataset.testExamples.count)
