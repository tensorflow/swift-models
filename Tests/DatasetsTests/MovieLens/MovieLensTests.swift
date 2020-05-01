import Datasets
import Foundation
import TensorFlow
import XCTest

final class MovieLensTests: XCTestCase {
    func testCreateMovieLens() {
        let dataset = MovieLens()
        XCTAssertEqual(dataset.numUsers, 943)
        XCTAssertEqual(dataset.numItems, 1650)
        XCTAssertEqual(dataset.trainMatrix.count, 400000)
    }
}

extension MovieLensTests {
    static var allTests = [
        ("testCreateMovieLens", testCreateMovieLens)
    ]
}
