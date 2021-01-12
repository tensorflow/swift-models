import Datasets
import Foundation
import TensorFlow
import XCTest

final class CIFAR100Tests: XCTestCase {
    func testCreateCIFAR100() {
        let dataset = CIFAR100(
            batchSize: 1,
            entropy: SystemRandomNumberGenerator(),
            device: Device.default,
            remoteBinaryArchiveLocation:
                URL(
                    string:
                        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
                )!, normalizing: true
        )
        verify(dataset)
    }

    func verify(_ dataset: CIFAR100<SystemRandomNumberGenerator>) {
        var totalCount = 0
        for epochBatches in dataset.training.prefix(1){ 
            for batch in epochBatches {
                XCTAssertTrue((0..<100).contains(batch.label[0].scalar!))
                XCTAssertEqual(batch.data.shape, [1, 32, 32, 3])
                totalCount += 1
            }
        }
        XCTAssertEqual(totalCount, 50000)
    }
    
    func testNormalizeCIFAR100() {
        let dataset = CIFAR100(
            batchSize: 50000,
            entropy: SystemRandomNumberGenerator(),
            device: Device.default,
            remoteBinaryArchiveLocation:
                URL(
                    string:
                        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
                )!, normalizing: true
        )
        
        let targetMean = Tensor<Double>([0, 0, 0])
        let targetStd = Tensor<Double>([1, 1, 1])
        for epochBatches in dataset.training.prefix(1){ 
            for batch in epochBatches {
                let images = Tensor<Double>(batch.data)
                let mean = images.mean(squeezingAxes: [0, 1, 2])
                let std = images.standardDeviation(squeezingAxes: [0, 1, 2])
                XCTAssertTrue(targetMean.isAlmostEqual(to: mean,
                                                       tolerance: 1e-3))
                XCTAssertTrue(targetStd.isAlmostEqual(to: std,
                                                      tolerance: 1e-3))
            }
        }
    }
}

extension CIFAR100Tests {
    static var allTests = [
        ("testCreateCIFAR100", testCreateCIFAR100),
        ("testNormalizeCIFAR100", testNormalizeCIFAR100),
    ]
}