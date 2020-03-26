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
            batchSize: 1,
            remoteBinaryArchiveLocation:
                URL(
                    string:
                        "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz"
                )!, normalizing: true
        )
        verify(dataset)
    }

    func verify(_ dataset: CIFAR10) {
        var totalCount = 0
        for example in dataset.trainingBatcher.sequenced() {
            XCTAssertTrue((0..<10).contains(example.second[0].scalar!))
            XCTAssertEqual(example.first.shape, [1, 32, 32, 3])
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