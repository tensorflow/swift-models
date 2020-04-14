import Datasets
import Foundation
import TensorFlow
import XCTest

final class COCOTests: XCTestCase {
    func testLoadCaptionsTrain2017() {
        let coco = COCOVariant.loadCaptionsTrain2017()
        verify(coco)
    }

    func testLoadCaptionsVal2017() {
        let coco = COCOVariant.loadCaptionsVal2017()
        verify(coco)
    }

    func testLoadInstancesTrain2017() {
        let coco = COCOVariant.loadInstancesTrain2017()
        verify(coco)
    }

    func testLoadInstancesVal2017() {
        let coco = COCOVariant.loadInstancesVal2017()
        verify(coco)
    }

    func testLoadPersonKeypointsTrain2017() {
        let coco = COCOVariant.loadPersonKeypointsTrain2017()
        verify(coco)
    }

    func testLoadPersonKeypointsVal2017() {
        let coco = COCOVariant.loadPersonKeypointsVal2017()
        verify(coco)
    }

    func verify(_ coco: COCO) {
        assert(coco.images.count > 0)
        assert(coco.annotations.count > 0)
    }

    static var allTests = [
        ("testLoadCaptionsTrain2017", testLoadCaptionsTrain2017),
        ("testLoadCaptionsVal2017", testLoadCaptionsVal2017),
        ("testLoadInstancesTrain2017", testLoadInstancesTrain2017),
        ("testLoadInstancesVal2017", testLoadInstancesVal2017),
        ("testLoadPersonKeypointsTrain2017", testLoadPersonKeypointsTrain2017),
        ("testLoadPersonKeypointsVal2017", testLoadPersonKeypointsVal2017),
    ]
}
