import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCODatasetTests: XCTestCase {
    func testExamplesNoMasks() {
        let dataset = COCODataset(batchSize: 32, includeMasks: false)
        XCTAssertTrue(dataset.trainingExamples.count > 0)
        XCTAssertTrue(dataset.testExamples.count > 0)
    }

    func testExamplesIncludingMasks() {
        let dataset = COCODataset(batchSize: 32, includeMasks: true)
        XCTAssertTrue(dataset.trainingExamples.count > 0)
        XCTAssertTrue(dataset.testExamples.count > 0)
    }

    static var allTests = [
        ("testExamplesNoMasks", testExamplesNoMasks),
        ("testExamplesIncludingMasks", testExamplesIncludingMasks),
    ]
}
