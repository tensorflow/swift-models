import TensorFlow
import XCTest
import Datasets

final class MNISTTests: XCTestCase {
    func testCreateMNIST() {
        let dataset = MNIST(batchSize: 1)

        var totalCount = 0
        for epochBatches in dataset.training.prefix(1){ 
            for batch in epochBatches {
                XCTAssertTrue((0..<10).contains(batch.label[0].scalar!))
                XCTAssertEqual(batch.data.shape, [1, 28, 28, 1])
                totalCount += 1
            }
        }
        XCTAssertEqual(totalCount, 60000)
    }

    func testCreateFashionMNIST() {
        let dataset = FashionMNIST(batchSize: 1)

        var totalCount = 0
        for epochBatches in dataset.training.prefix(1){ 
            for batch in epochBatches {
                XCTAssertTrue((0..<10).contains(batch.label[0].scalar!))
                XCTAssertEqual(batch.data.shape, [1, 28, 28, 1])
                totalCount += 1
            }
        }
        XCTAssertEqual(totalCount, 60000)
    }

    func testCreateKuzushijiMNIST() {
        let dataset = KuzushijiMNIST(batchSize: 1)

        var totalCount = 0
        for epochBatches in dataset.training.prefix(1){ 
            for batch in epochBatches {
                XCTAssertTrue((0..<10).contains(batch.label[0].scalar!))
                XCTAssertEqual(batch.data.shape, [1, 28, 28, 1])
                totalCount += 1
            }
        }
        XCTAssertEqual(totalCount, 60000)
    }
}

extension MNISTTests {
    static var allTests = [
        ("testCreateMNIST", testCreateMNIST),
        ("testCreateFashionMNIST", testCreateFashionMNIST),
        ("testCreateKuzushijiMNIST", testCreateKuzushijiMNIST),
    ]
}

