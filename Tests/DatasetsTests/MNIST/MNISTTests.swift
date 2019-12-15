import TensorFlow
import XCTest
import Datasets

final class MNISTTests: XCTestCase {
    func testCreateMNIST() {
        let dataset = MNIST()

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertTrue((0..<10).contains(example.label.scalar!))
            XCTAssertEqual(example.data.shape, [28, 28, 1])
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 60000)
    }
}

extension MNISTTests {
    static var allTests = [
        ("testCreateMNIST", testCreateMNIST),
    ]
}

