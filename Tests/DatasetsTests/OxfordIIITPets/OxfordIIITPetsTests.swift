import Datasets
import TensorFlow
import XCTest

final class OxfordIIITPetsTests: XCTestCase {
  func testCreateOxfordIIITPets() {
    let dataset = OxfordIIITPets(batchSize: 64)

    var batchCount = 0
    for epochBatches in dataset.training.prefix(1) {
      for batch in epochBatches {
        batchCount += 1
        /// 3680 samples make 57 batches of size 64 and one batch of size 32
        let expectedBS = batchCount <= 57 ? 64 : 32

        XCTAssertEqual(batch.data.shape, [expectedBS, 224, 224, 3])
        XCTAssertEqual(batch.label.shape, [expectedBS, 224, 224, 1])
      }
    }
    XCTAssertEqual(batchCount, 57)
  }
}

extension OxfordIIITPetsTests {
  static var allTests = [
    ("testCreateOxfordIIITPets", testCreateOxfordIIITPets),
  ]
}
