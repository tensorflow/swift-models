import Foundation
import TensorFlow
import PythonKit

func generateAndVisualize(m: HandwritingGenerator, text: String, bias: Float) {
    let iSeqPt = Tensor<Float>([[0, 0, 1]])
    let iSeqStr = Tensor("\(text) ".map { Int32(HandwritingGenerator.charDict[$0]!) }).reshaped(to: [-1, 1])
    let iSeqStrMask = Tensor<Float>(ones: iSeqStr.shape)
    let (seqPt, _) = m.predict(ptIni: iSeqPt, seqStr: iSeqStr, seqStrMask: iSeqStrMask, bias: bias)

    plt.figure(figsize: [10, 2])
    plt.savefig("Handwriting.png")
    visualize(strokes: seqPt.squeezingShape(at: 1))
}

generateAndVisualize(m: loadModel(),
                     text: CommandLine.arguments[1],
                     bias: Float(CommandLine.arguments[2])!)
