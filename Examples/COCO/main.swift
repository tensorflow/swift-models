import Datasets
import Foundation
import SwiftProtobuf

var dataset = COCODataset(includeMasks: true, batchSize: 128, numWorkers: 72)
print(dataset.trainingExamples[0].image.tensor().shape)
