import Python
import TensorFlow

let sys = Python.import("sys")
sys.path = sys.path + ["."]
let encoder = Python.import("encoder").get_encoder("117M")

let checkpoint = "models/117M/model.ckpt"
let model = TransformerLM(contentsOfPythonCheckpointFile: checkpoint, scope: "model")

let start_token = Int32(encoder.encoder["<|endoftext|>"])!
var tokens = Tensor(shape: [1, 1], scalars: [start_token])
var temperature = Float(1.0)

if CommandLine.arguments.count >= 2 {
    temperature = Float(CommandLine.arguments[1])!
}

if CommandLine.arguments.count == 3 {
    let seed = CommandLine.arguments[2]
    print(seed, terminator: "")
    let pytok = encoder.encode(seed)
    let tokarr: [Int32] = Array<Int>(pytok)!.map { Int32($0) }
    tokens = Tensor(shape: [1, tokarr.count], scalars: tokarr)
}

let empty = Tensor<Float>(zeros: [config.headCount, 0, config.embeddingSize / config.headCount])
var states = (0..<config.layerCount).map { _ in AttentionContext(key: empty, value: empty) }

for _ in 0..<100 {
    let logits = model(tokens, states: &states)
    let (batchSize, timeSteps, vocabSize) = (logits.shape[0], logits.shape[1], logits.shape[2])
    let lastLogit = logits.slice(
        lowerBounds: [0, timeSteps - 1, 0],
        upperBounds: [batchSize, timeSteps, vocabSize]) / temperature
    tokens = Raw.multinomial(logits: lastLogit.squeezingShape(at: 1), numSamples: Tensor<Int32>(1))
    print(encoder.decode(tokens[0].makeNumpyArray()), terminator: "")
}
print()
