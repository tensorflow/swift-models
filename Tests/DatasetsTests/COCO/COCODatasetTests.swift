import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCODatasetTests: XCTestCase {
    func testExamplesNoMasks() {
        // We use val/test variants here, instead of train/val, 
        // to avoid fetching the full training data during CI runs.
        let dataset = COCODataset(
            training: COCOVariant.loadVal(),
            validation: COCOVariant.loadTest(),
            includeMasks: false, batchSize: 32)
      
        for epochBatches in dataset.training.prefix(1) {
          let batch = epochBatches.first!
          XCTAssertTrue(batch[0].image.width != 0)
        }

        let validationBatch = dataset.validation.first!
        XCTAssertTrue(validationBatch[0].image.width != 0)
    }

    func testExamplesIncludingMasks() {
        // We use val/test variants here, instead of train/val, 
        // to avoid fetching the full training data during CI runs.
        let dataset = COCODataset(
            training: COCOVariant.loadVal(),
            validation: COCOVariant.loadTest(),
            includeMasks: true, batchSize: 32)

        for epochBatches in dataset.training.prefix(1) {
          let batch = epochBatches.first!
          batch[0].image.width != 0
        }

        let validationBatch = dataset.validation.first!
        XCTAssertTrue(validationBatch[0].image.width != 0)
    }

    static var allTests = [
        ("testExamplesNoMasks", testExamplesNoMasks),
        ("testExamplesIncludingMasks", testExamplesIncludingMasks),
    ]
}
