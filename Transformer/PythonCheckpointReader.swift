import TensorFlow

struct Config {
    let vocabSize: Int
    let contextSize: Int
    let embeddingSize: Int
    let headCount: Int
    let layerCount: Int
}

extension Config {
    init(dict: [String: Int]) {
        vocabSize = dict["n_vocab"]!
        contextSize = dict["n_ctx"]!
        embeddingSize = dict["n_embd"]!
        headCount = dict["n_head"]!
        layerCount = dict["n_layer"]!
    }
}

let config = Config(dict: [
    "n_vocab": 50257,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12])

func readTensor<Scalar: TensorFlowScalar>(
    from path: String, name: String, scalarType: Scalar.Type
) -> Tensor<Scalar> {
    // TODO(jekbradbury): support variadic dtype attrs in RawOpsGenerated
    return Tensor<Scalar>(handle: #tfop(
        "RestoreV2",
        StringTensor(path),
        StringTensor([name]),
        StringTensor([""]),
        dtypes$dtype: [Scalar.tensorFlowDataType]))
}

private func checkShapes(_ tensor1: Tensor<Float>, _ tensor2: Tensor<Float>) {
    guard tensor1.shape == tensor2.shape else {
        print("Shape mismatch: \(tensor1.shape) != \(tensor2.shape)")
        fatalError()
    }
}

protocol InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String)
}

extension Dense: InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String) {
        let kernel = readTensor(from: path, name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: readTensor(from: path, name: scope + "/b", scalarType: Scalar.self),
            activation: identity)
    }
    init(from path: String, withScope scope: String, activation: String) {
        let kernel = readTensor(from: path, name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: readTensor(from: path, name: scope + "/b", scalarType: Scalar.self),
            activation: gelu)
    }
}

extension LayerNorm: InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String) {
        self.init(
            offset: readTensor(from: path, name: scope + "/b", scalarType: Scalar.self),
            scale: readTensor(from: path, name: scope + "/g", scalarType: Scalar.self),
            axis: -1,
            epsilon: Tensor(1e-5))
    }
}

extension MultiHeadAttention: InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String) {
        attention = Attention(size: config.embeddingSize / config.headCount, causal: true, dropProbability: 0.2)
        wqkv = TimeDistributed(Dense<Float>(from: path, withScope: scope + "/c_attn"))
        wo = TimeDistributed(Dense<Float>(from: path, withScope: scope + "/c_proj"))
        headCount = Int32(12)
    }
}

extension FeedForward: InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String) {
        dense1 = TimeDistributed(Dense<Float>(from: path, withScope: scope + "/c_fc", activation: "gelu"))
        dense2 = TimeDistributed(Dense<Float>(from: path, withScope: scope + "/c_proj"))
        dropout = Dropout(probability: 0.2)
    }
}

extension EncoderLayer: InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String) {
        selfAttention = MultiHeadAttention(from: path, withScope: scope + "/attn")
        selfAttentionDropout = Dropout(probability: 0.2)
        selfAttentionNorm = LayerNorm(from: path, withScope: scope + "/ln_1")
        feedForward = FeedForward(from: path, withScope: scope + "/mlp")
        feedForwardDropout = Dropout(probability: 0.2)
        feedForwardNorm = LayerNorm(from: path, withScope: scope + "/ln_2")
    }
}

extension TransformerLM: InitializableFromPythonCheckpoint {
    init(from path: String, withScope scope: String) {
        embedding = Embedding(
            weight: readTensor(from: path, name: scope + "/wte", scalarType: Float.self))
        positionalEmbeddings = readTensor(from: path, name: scope + "/wpe", scalarType: Float.self)
        layers = (0..<config.layerCount).map { i in
            EncoderLayer(from: path, withScope: scope + "/h\(i)")
        }
        norm = LayerNorm(from: path, withScope: scope + "/ln_f")
    }
}
