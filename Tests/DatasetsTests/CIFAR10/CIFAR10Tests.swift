import Datasets
import Foundation
import TensorFlow
import XCTest

final class CIFAR10Tests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Force downloading of the dataset during tests by removing any pre-existing local files.
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(
            "CIFAR10/cifar-10-batches-bin").path
        if FileManager.default.fileExists(atPath: path) {
            try! FileManager.default.removeItem(atPath: path)
        }
    }

    func testCreateCIFAR10() {
        let dataset = CIFAR10(
            remoteBinaryArchiveLocation:
                URL(
                    string:
                        "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz"
                )!
        )
        verify(dataset)
    }

    func verify(_ dataset: CIFAR10) {
        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertTrue((0..<10).contains(example.label.scalar!))
            XCTAssertEqual(example.data.shape, [32, 32, 3])
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 50000)
    }
}

extension CIFAR10Tests {
    static var allTests = [
        ("testCreateCIFAR10", testCreateCIFAR10),
    ]
}