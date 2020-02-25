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
    
    func testImagenetteBatchers() {
        let batchers = ImagenetteBatchers()
        
        var batchCount = 0
        for batch in batchers.training.sequenced() {
            batchCount += 1
            //12894 samples make 201 batches of size 64 and one last batch of size 30
            let expectedBS = batchCount <= 201 ? 64 : 30
            
            XCTAssertEqual(batch.data.shape, [expectedBS, 224, 224, 3])
        }
    }
}

extension ImagenetteTests {
    static var allTests = [
        ("testCreateImagenette", testCreateImagenette),
        ("testImagenetteBatchers", testImagenetteBatchers),
        ("testCreateImagewoof", testCreateImagewoof),
    ]
}

