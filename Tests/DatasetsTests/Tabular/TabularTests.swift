import Datasets
import Foundation
import TensorFlow
import XCTest

final class TabularTests: XCTestCase {
    func testCreateBostonHousing() {
        let dataset = BostonHousing()
        XCTAssertEqual(dataset.numRecords, 506)
    }
}

extension TabularTests {
    static var allTests = [
        ("testCreateBostonHousing", testCreateBostonHousing),
    ]
}