import Datasets
import Foundation
import TensorFlow
import XCTest

final class TabularTests: XCTestCase {
    func testCreateBostonHousing() {
        let dataset = BostonHousing()
        XCTAssertEqual(dataset.numRecords, 506)
        XCTAssertEqual(dataset.numTrainRecords, 405)
        XCTAssertEqual(dataset.numTestRecords, 101)
    }
}

extension TabularTests {
    static var allTests = [
        ("testCreateBostonHousing", testCreateBostonHousing),
    ]
}