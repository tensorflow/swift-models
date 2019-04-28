import TensorFlow
import XCTest

@testable import MiniGo

final class GoModelTests: XCTestCase {
    func testInferenceShape() {
        let modelConfiguration = ModelConfiguration(boardSize: 19)
        let model = GoModel(configuration: modelConfiguration)

        let sampleInput = Tensor<Float>(randomUniform: [2, 19, 19, 17])
        let inference = model.prediction(for: sampleInput)
        let (policy, value) = (inference.policy, inference.value)
        XCTAssertEqual(TensorShape([2, 19 * 19 + 1]), policy.shape)
        XCTAssertEqual(TensorShape([2]), value.shape)
    }
}

extension GoModelTests {
    static var allTests = [
        ("testInferenceShape", testInferenceShape),
    ]
}
