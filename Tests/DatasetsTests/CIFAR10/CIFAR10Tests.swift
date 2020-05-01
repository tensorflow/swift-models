import Datasets
import Foundation
import TensorFlow
import XCTest

final class CIFAR10Tests: XCTestCase {
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
        for example in dataset.training.sequenced() {
            XCTAssertTrue((0..<10).contains(example.second[0].scalar!))
            XCTAssertEqual(example.first.shape, [1, 32, 32, 3])
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 50000)
    }

    func testNormalizeCIFAR10() {
        let dataset = CIFAR10(
            batchSize: 50000,
            remoteBinaryArchiveLocation:
                URL(
                    string:
                        "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz"
                )!, normalizing: true
        )

        let targetMean = Tensor<Double>([0, 0, 0])
        let targetStd = Tensor<Double>([1, 1, 1])
        for batch in dataset.training.sequenced() {
            let images = Tensor<Double>(batch.first)
            let mean = images.mean(squeezingAxes: [0, 1, 2])
            let std = images.standardDeviation(squeezingAxes: [0, 1, 2])
            XCTAssertTrue(
                targetMean.isAlmostEqual(
                    to: mean,
                    tolerance: 1e-6))
            XCTAssertTrue(
                targetStd.isAlmostEqual(
                    to: std,
                    tolerance: 1e-5))
        }
    }
}

extension CIFAR10Tests {
    static var allTests = [
        ("testCreateCIFAR10", testCreateCIFAR10),
        ("testNormalizeCIFAR10", testNormalizeCIFAR10),
    ]
}
