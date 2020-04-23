import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCOVariantTests: XCTestCase {
    func testLoadTrain() {
        let coco = COCOVariant.loadTrain()
        assert(coco.images.count > 0)
        assert(coco.annotations.count > 0)
    }

    func testLoadVal() {
        let coco = COCOVariant.loadVal()
        assert(coco.images.count > 0)
        assert(coco.annotations.count > 0)
    }

    func testLoadTest() {
        let coco = COCOVariant.loadTest()
        assert(coco.images.count > 0)
        assert(coco.annotations.count == 0)
    }

    func testLoadTestDev() {
        let coco = COCOVariant.loadTestDev()
        assert(coco.images.count > 0)
        assert(coco.annotations.count == 0)
    }

    static var allTests = [
        // We exclude full train dataset to avoid downloading
        // too much data during the swift-models CI runs.
        // ("testLoadTestTrain", testLoadTrain),
        ("testLoadTestVal", testLoadVal),
        ("testLoadTest", testLoadTest),
        ("testLoadTestDev", testLoadTestDev),
    ]
}
