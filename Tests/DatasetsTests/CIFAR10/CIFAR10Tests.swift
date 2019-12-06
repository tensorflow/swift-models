import TensorFlow
import XCTest

@testable import Datasets

final class CIFAR10Tests: XCTestCase {
	func testCreateCIFAR10() {
		let dataset = CIFAR10()

		var totalCount = 0
		for example in dataset.trainingDataset {
	    	XCTAssertTrue((0..<10).contains(example.label.scalar!))
	    	XCTAssertEqual(example.data.shape, [32, 32, 3])
			totalCount += 1
		}
		XCTAssertEqual(totalCount, 50000)
	}
}
