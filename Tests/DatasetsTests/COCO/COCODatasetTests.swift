import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCODatasetTests: XCTestCase {
    func testExamplesNoMasks() {
        let dataset = COCODataset(includeMasks: false, batchSize: 32, numWorkers: 8)
        XCTAssertTrue(dataset.trainingExamples.count > 0)
        XCTAssertTrue(dataset.testExamples.count > 0)
    }

    func testExamplesIncludingMasks() {
        let dataset = COCODataset(includeMasks: true, batchSize: 32, numWorkers: 8)
        XCTAssertTrue(dataset.trainingExamples.count > 0)
        XCTAssertTrue(dataset.testExamples.count > 0)
    }

    static var allTests = [
        ("testExamplesNoMasks", testExamplesNoMasks),
        ("testExamplesIncludingMasks", testExamplesIncludingMasks),
    ]
}
