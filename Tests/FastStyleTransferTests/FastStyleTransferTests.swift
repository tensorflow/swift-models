import XCTest
@testable import FastStyleTransfer
import TensorFlow

final class FastStyleTransferTests: XCTestCase {
    func testSanity() {
        let model = TransformerNet()
        let tmp = Tensor<Float>(randomUniform: [3, 200, 320, 3])
        let out = model(tmp)
        XCTAssertEqual(out.shape, [3, 200, 320, 3])
    }

    static var allTests = [
        ("testSanity", testSanity),
    ]
}
