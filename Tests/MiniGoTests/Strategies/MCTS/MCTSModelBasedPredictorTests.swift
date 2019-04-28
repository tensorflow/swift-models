import TensorFlow
import XCTest

@testable import MiniGo

final class MCTSModelBasedPredictorTests: XCTestCase {

    struct MockModel: InferenceModel {
        func prediction(for: Tensor<Float>) -> GoModelOutput {
            return GoModelOutput(
                policy: Tensor<Float>(rangeFrom: 0, to: 19 * 19 + 1, stride:1),
                value: Tensor<Float>(shape: [1], scalars: [0.9]),
                logits: Tensor<Float>(randomUniform: [1, 19 * 19 + 1]))  // Not used.
        }
    }

    func testPrediction() {
        let configuration = GameConfiguration(size: 19, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration)

        let predictor = MCTSModelBasedPredictor(boardSize: 19, model: MockModel())
        let prediction = predictor.prediction(for: boardState)

        XCTAssertEqual(0.9, prediction.rewardForNextPlayer)
        XCTAssertEqual(Float(19 * 19), prediction.distribution.pass)
        for x in 0..<19 {
            for y in 0..<19 {
                XCTAssertEqual(ShapedArraySlice(Float(x * 19 + y)), prediction.distribution.positions[x][y])
            }
        }
    }
}

extension MCTSModelBasedPredictorTests {
    static var allTests = [
        ("testPrediction", testPrediction),
    ]
}
