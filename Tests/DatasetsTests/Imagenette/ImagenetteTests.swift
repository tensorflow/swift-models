import Datasets
import TensorFlow
import XCTest

final class ImagenetteTests: XCTestCase {
    func testCreateImagenette() {
        let dataset = Imagenette(batchSize: 64)

        var batchCount = 0
        for batch in dataset.training.sequenced() {
            batchCount += 1
            //12894 samples make 201 batches of size 64 and one last batch of size 30
            let expectedBS = batchCount <= 201 ? 64 : 30

            XCTAssertEqual(batch.first.shape, [expectedBS, 224, 224, 3])
        }
        XCTAssertEqual(batchCount, dataset.training.count)
    }

    func testCreateImagewoof() {
        let dataset = Imagewoof(batchSize: 64)

        var batchCount = 0
        for batch in dataset.training.sequenced() {
            batchCount += 1
            //12454 samples make 194 batches of size 64 and one last batch of size 38
            let expectedBS = batchCount <= 194 ? 64 : 38

            XCTAssertEqual(batch.first.shape, [expectedBS, 224, 224, 3])
        }
        XCTAssertEqual(batchCount, dataset.training.count)
    }
}

extension ImagenetteTests {
    static var allTests = [
        ("testCreateImagenette", testCreateImagenette),
        ("testCreateImagewoof", testCreateImagewoof),
    ]
}
