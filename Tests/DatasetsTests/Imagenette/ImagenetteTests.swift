import TensorFlow
import XCTest
import Datasets

final class ImagenetteTests: XCTestCase {
    func testCreateImagenette() {
        let dataset = Imagenette()

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertEqual(example.data.shape, [224, 224, 3])
            totalCount += 1
        }
        XCTAssertEqual(totalCount, dataset.trainingExampleCount)
    }
	
    func testCreateImagewoof() {
        let dataset = Imagewoof()

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertEqual(example.data.shape, [224, 224, 3])
            totalCount += 1
        }
        XCTAssertEqual(totalCount, dataset.trainingExampleCount)
    }
}

extension ImagenetteTests {
    static var allTests = [
        ("testCreateImagenette", testCreateImagenette),
        ("testCreateImagewoof", testCreateImagewoof),
    ]
}

