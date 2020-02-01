import Datasets
import Foundation
import TensorFlow
import XCTest

final class MNISTTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Force downloading of the dataset during tests by removing any pre-existing local files.
        let mnistFileNames = [
            "train-images-idx3-ubyte"
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte"
        ]
        for filename in  mnistFileNames {
            let path = FileManager.default.temporaryDirectory.appendingPathComponent(
                "MNIST/\(filename)").path
            if FileManager.default.fileExists(atPath: path) {
                try! FileManager.default.removeItem(atPath: path)
            }
        }
    }

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
        ("testCreateMNIST", testCreateMNIST)
    ]
}
