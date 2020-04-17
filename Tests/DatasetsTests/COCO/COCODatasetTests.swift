import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCODatasetTests: XCTestCase {
    func testExamplesNoMasks() {
        let dataset = COCODataset(includeMasks: false, batchSize: 32, numWorkers: 8)
        verify(dataset.trainingExamples)
        verify(dataset.testExamples)
    }

    func testExamplesIncludingMasks() {
        let dataset = COCODataset(includeMasks: true, batchSize: 32, numWorkers: 8)
        verify(dataset.trainingExamples)
        verify(dataset.testExamples)
    }

    func verify(_ examples: [ObjectDetectionExample]) {
        XCTAssertTrue(examples.count > 0)
        XCTAssertTrue(examples[0].image.tensor().shape.count > 0)
    }

    static var allTests = [
        ("testExamplesNoMasks", testExamplesNoMasks),
        ("testExamplesIncludingMasks", testExamplesIncludingMasks),
    ]
}
