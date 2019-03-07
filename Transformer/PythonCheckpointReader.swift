import TensorFlow

struct Config {
    let vocabSize: Int
    let contextSize: Int
    let embeddingSize: Int
    let headCount: Int
    let layerCount: Int
}

extension Config {
    init(dictionary: [String: Int]) {
        vocabSize = dictionary["n_vocab"]!
        contextSize = dictionary["n_ctx"]!
        embeddingSize = dictionary["n_embd"]!
        headCount = dictionary["n_head"]!
        layerCount = dictionary["n_layer"]!
    }
}

let config = Config(dictionary: [
    "n_vocab": 50257,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12
])

func readTensor<Scalar: TensorFlowScalar>(
    fromPath path: String,
    name: String,
    scalarType: Scalar.Type
) -> Tensor<Scalar> {
    // TODO(jekbradbury): support variadic dtype attrs in RawOpsGenerated
    return Tensor(handle: #tfop(
        "RestoreV2",
        StringTensor(path),
        StringTensor([name]),
        StringTensor([""]),
        dtypes$dtype: [Scalar.tensorFlowDataType]))
}

protocol InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String)
}

extension Dense: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String) {
        let kernel = readTensor(fromPath: path, name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: readTensor(fromPath: path, name: scope + "/b", scalarType: Scalar.self),
            activation: identity)
    }
    init(contentsOfPythonCheckpointFile path: String, scope: String, activation: String) {
        let kernel = readTensor(fromPath: path, name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: readTensor(fromPath: path, name: scope + "/b", scalarType: Scalar.self),
            activation: gelu)
    }
}

extension LayerNorm: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String) {
        self.init(
            offset: readTensor(fromPath: path, name: scope + "/b", scalarType: Scalar.self),
            scale: readTensor(fromPath: path, name: scope + "/g", scalarType: Scalar.self),
            axis: -1,
            epsilon: Tensor(1e-5))
    }
}

extension MultiHeadAttention: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String) {
        attention = Attention(
            size: config.embeddingSize / config.headCount,
            causal: true,
            dropProbability: 0.2)
        wqkv = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            scope: scope + "/c_attn"))
        wo = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            scope: scope + "/c_proj"))
        headCount = Int32(12)
    }
}

extension FeedForward: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String) {
        dense1 = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            scope: scope + "/c_fc", activation: "gelu"))
        dense2 = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            scope: scope + "/c_proj"))
        dropout = Dropout(probability: 0.2)
    }
}

extension EncoderLayer: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String) {
        selfAttention = MultiHeadAttention(
            contentsOfPythonCheckpointFile: path,
            scope: scope + "/attn")
        selfAttentionDropout = Dropout(probability: 0.2)
        selfAttentionNorm = LayerNorm(contentsOfPythonCheckpointFile: path, scope: scope + "/ln_1")
        feedForward = FeedForward(contentsOfPythonCheckpointFile: path, scope: scope + "/mlp")
        feedForwardDropout = Dropout(probability: 0.2)
        feedForwardNorm = LayerNorm(contentsOfPythonCheckpointFile: path, scope: scope + "/ln_2")
    }
}

extension TransformerLM: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, scope: String) {
        embedding = Embedding(
            weight: readTensor(fromPath: path, name: scope + "/wte", scalarType: Float.self))
        positionalEmbeddings = readTensor(
            fromPath: path,
            name: scope + "/wpe",
            scalarType: Float.self)
        layers = (0..<config.layerCount).map { i in
            EncoderLayer(contentsOfPythonCheckpointFile: path, scope: scope + "/h\(i)")
        }
        norm = LayerNorm(contentsOfPythonCheckpointFile: path, scope: scope + "/ln_f")
    }
}
