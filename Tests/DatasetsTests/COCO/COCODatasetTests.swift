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
            test: COCOVariant.loadTest(),
            includeMasks: false, batchSize: 32, numWorkers: 8)
        verify(dataset.trainingExamples)
        verify(dataset.testExamples)
    }

    func testExamplesIncludingMasks() {
        // We use val/test variants here, instead of train/val, 
        // to avoid fetching the full training data during CI runs.
        let dataset = COCODataset(
            training: COCOVariant.loadVal(),
            test: COCOVariant.loadTest(),
            includeMasks: true, batchSize: 32, numWorkers: 8)
        verify(dataset.trainingExamples)
        verify(dataset.testExamples)
    }

    func verify(_ examples: [ObjectDetectionExample]) {
        XCTAssertTrue(examples.count > 0)
        XCTAssertTrue(examples[0].image.width != 0)
    }

    static var allTests = [
        ("testExamplesNoMasks", testExamplesNoMasks),
        ("testExamplesIncludingMasks", testExamplesIncludingMasks),
    ]
}
