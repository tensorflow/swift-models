import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCOVariantTests: XCTestCase {
    func testLoadTrain() {
        let coco = COCOVariant.loadTrain()
        verify(coco)
    }

    func testLoadVal() {
        let coco = COCOVariant.loadVal()
        verify(coco)
    }

    func verify(_ coco: COCO) {
        assert(coco.images.count > 0)
        assert(coco.annotations.count > 0)
    }

    static var allTests = [
        ("testLoadTrain", testLoadTrain),
        ("testLoadVal", testLoadVal),
    ]
}
