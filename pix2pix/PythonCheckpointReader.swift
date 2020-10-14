


import Checkpoints
import ModelSupport
import TensorFlow
import Foundation

public struct NetGConfig: Codable {
    public let inChannels: Int
    public let outChannels: Int
    public let ngf: Int  // size of feature map
    public let useDropout: Bool
    public let lastConvFilters: Int
    public let learningRate: Float
    public let beta: Float
    public let padding: Int?
    public let kernelSize: Int
    
    enum CodingKeys: String, CodingKey {
        case inChannels = "i_channels"
        case outChannels = "o_channels"
        case ngf = "ngf"
        case useDropout = "useDrop"
        case lastConvFilters = "n_lastConvFilters"
        case learningRate = "lRate"
        case beta = "beta"
        case padding = "pad"
        case kernelSize = "kSize"
    }
}

extension CheckpointReader {
    func readTensor<Scalar: TensorFlowScalar>(
        name: String
    ) -> Tensor<Scalar> {
        return Tensor<Scalar>(loadTensor(named: name))
    }
    func readIntTensor<Int: TensorFlowInteger>(
        name: String
    ) -> Tensor<Int> {
        return Tensor<Int>(loadTensor(named: name))
    }
}

// TODO: Come up with better names for these protocols
protocol InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String)
}

protocol InitializableFromPythonCheckpoint3 {
    associatedtype Sublayer: Layer where Sublayer.TangentVector.VectorSpaceScalar == Float, Sublayer.Input == Tensor<Float>, Sublayer.Output == Tensor<Float> 
    init(reader: CheckpointReader, config: NetGConfig, scope: String, submodule: Sublayer)
}

//extension Dense: InitializableFromPythonCheckpoint {
//    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
//        var kernel: Tensor<Scalar> = reader.readTensor(name: scope  "/w")
//        if kernel.shape.dimensions.count > 2 {
//            // The OpenAI checkpoints have a batch dimension, and our checkpoints do not.
//            kernel = kernel.squeezingShape(at: 0)
//        }
//        self.init(
//            weight: kernel,
//            bias: reader.readTensor(name: scope  "/b"),
//            activation: identity)
//    }
//
//    init(
//        reader: CheckpointReader,
//        config: TransformerLMConfig,
//        scope: String,
//        activation: String
//    ) {
//        var kernel: Tensor<Scalar> = reader.readTensor(name: scope  "/w")
//        if kernel.shape.dimensions.count > 2 {
//            // The OpenAI checkpoints have a batch dimension, and our checkpoints do not.
//            kernel = kernel.squeezingShape(at: 0)
//        }
//        self.init(
//            weight: kernel,
//            bias: reader.readTensor(name: scope  "/b"),
//            activation: gelu)
//    }
//}
//
//extension LayerNorm: InitializableFromPythonCheckpoint {
//    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
//        self.init(
//            offset: reader.readTensor(name: scope  "/b"),
//            scale: reader.readTensor(name: scope  "/g"),
//            axis: -1,
//            epsilon: 1e-5)
//    }
//}
//
//extension FeedForward: InitializableFromPythonCheckpoint {
//    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
//        dense1 = TimeDistributed(
//            Dense<Float>(reader: reader, config: config, scope: scope  "/c_fc", activation: "gelu")
//        )
//        dense2 = TimeDistributed(
//            Dense<Float>(reader: reader, config: config, scope: scope  "/c_proj"))
//    }
//}
//
//extension EncoderLayer: InitializableFromPythonCheckpoint {
//    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
//        selfAttention = MultiHeadAttentionGPT2(
//            reader: reader, config: config, scope: scope  "/attn")
//        selfAttentionDropout = Dropout(probability: 0.1)
//        selfAttentionNorm = LayerNorm(reader: reader, config: config, scope: scope  "/ln_1")
//        feedForward = FeedForward(reader: reader, config: config, scope: scope  "/mlp")
//        feedForwardDropout = Dropout(probability: 0.1)
//        feedForwardNorm = LayerNorm(reader: reader, config: config, scope: scope  "/ln_2")
//    }
//}
//


extension ConvLayer: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        conv2d = Conv2D<Float>(reader: reader, config: config, scope: scope + "/conv2d")
        let padding = config.padding
        let _padding =  padding ?? Int(config.kernelSize / 2)
        pad = ZeroPadding2D(padding: ((_padding, _padding), (_padding, _padding)))
    }
}

extension Conv2D: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let filter: Tensor<Scalar> = reader.readTensor(name: scope + "/fil")
        let bias: Tensor<Scalar> = reader.readTensor(name: scope + "/bias")
//        let activation: Tensor<Scalar> = reader.readTensor(name: scope + "/act")
//        let strides: Tensor<Scalar> = reader.readTensor(name: scope + "/str")
//        let padding: Tensor<Scalar> = reader.readTensor(name: scope + "/pad")
//        let dialations: Tensor<Scalar> = reader.readTensor(name: scope + "dia")
        self.init(filter: filter, bias: bias, strides: (2,2), padding: .same)
    }
}

extension BatchNorm: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let axis: Tensor<Scalar> = reader.readTensor(name: scope + "/axis")
        let momentum: Tensor<Scalar> = reader.readTensor(name: scope + "/mom")
        let epsilon: Tensor<Scalar> = reader.readTensor(name: scope + "/eps")
        let axisVals = axis.array.scalars
        let axisValScalar = axisVals[0]
        let axisVal = Int(axisValScalar)
        let momentumVals = momentum.array.scalars
        let momentumVal = momentumVals[0]
        let epsilonVals = epsilon.array.scalars
        let epsilonVal = epsilonVals[0]
        let offset: Tensor<Scalar> = reader.readTensor(name: scope + "/off")
        let scale: Tensor<Scalar> = reader.readTensor(name: scope + "/sc")
        let runningMean: Tensor<Scalar> = reader.readTensor(name: scope + "/rmean")
        let runningVariance: Tensor<Scalar> = reader.readTensor(name: scope + "/rvar")
        self.init(axis: axisVal, momentum: momentumVal, offset: offset, scale: scale, epsilon: epsilonVal, runningMean: runningMean, runningVariance: runningVariance)
    }
}

extension TransposedConv2D: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let filter: Tensor<Scalar> = reader.readTensor(name: scope + "/fil")
        let bias: Tensor<Scalar> = reader.readTensor(name: scope + "/bias")
//        let activation: Tensor<Scalar> = reader.readTensor(name: scope + "/act")
//        let strides: Tensor<Scalar> = reader.readTensor(name: scope + "/str")
//        let padding: Tensor<Scalar> = reader.readTensor(name: scope + "/pad")
        self.init(filter: filter, bias: bias, strides: (2,2), padding: .same)
    }
}

extension UNetSkipConnectionInnermost: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let downConv: Conv2D<Float> = Conv2D(reader: reader, config: config, scope: scope + "/dc")
        let upNorm: BatchNorm<Float> = BatchNorm(reader: reader, config: config, scope: scope + "/un")
        let upConv: TransposedConv2D<Float> = TransposedConv2D(reader: reader, config: config, scope: scope + "/uc")
        self.init(downConv: downConv, upConv: upConv, upNorm: upNorm)
    }
}

extension UNetSkipConnection: InitializableFromPythonCheckpoint3 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String, submodule: Sublayer) {
        let downConv: Conv2D<Float> = Conv2D(reader: reader, config: config, scope: scope + "/dc")
        let downNorm: BatchNorm<Float> = BatchNorm(reader: reader, config: config, scope: scope + "/dn")
        let upConv: TransposedConv2D<Float> = TransposedConv2D(reader: reader, config: config, scope: scope + "/uc")
        let upNorm: BatchNorm<Float> = BatchNorm(reader: reader, config: config, scope: scope + "/un")
        let dropOut: Dropout<Float> = Dropout(reader: reader, config: config, scope: scope + "/drop")
        self.init(downConv: downConv, downNorm: downNorm, upConv: upConv, upNorm: upNorm, dropOut: dropOut, submodule: submodule)
    }
}

extension UNetSkipConnectionOutermost: InitializableFromPythonCheckpoint3 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String, submodule: Sublayer) {
        let downConv: Conv2D<Float> = Conv2D(reader: reader, config: config, scope: scope + "/dc")
        let upConv: TransposedConv2D<Float> = TransposedConv2D(reader: reader, config: config, scope: scope + "/uc")
        self.init(downConv: downConv, upConv: upConv, submodule: submodule)
    }
}

extension Dropout: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let probability:Tensor<Float> = reader.readTensor(name: scope + "/prob")
        let val = probability[0].scalar
        self.init(probability: Double(val!))
    }
}

extension NetG: InitializableFromPythonCheckpoint2 {
    public init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let firstBlock = UNetSkipConnectionInnermost(reader: reader, config: config, scope: scope + "/module/submod/submod/submod/submod/submod/submod/submod")
        let module1 = UNetSkipConnection<UNetSkipConnectionInnermost>(reader: reader, config: config, scope: scope + "/module/submod/submod/submod/submod/submod/submod", submodule: firstBlock)
        let module2 = UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>(reader: reader, config: config, scope: scope + "/module/submod/submod/submod/submod/submod", submodule: module1)
        let module3 = UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>>(reader: reader, config: config, scope: scope + "/module/submod/submod/submod/submod", submodule: module2)
        let module4 = UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>>>(reader: reader, config: config, scope: scope + "/module/submod/submod/submod", submodule: module3)
        let module5 = UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>>>>(reader: reader, config: config, scope: scope + "/module/submod/submod", submodule: module4)
        let module6 = UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>>>>>(reader: reader, config: config, scope: scope + "/module/submod", submodule: module5)
        self.module = UNetSkipConnectionOutermost<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>>>>>>(reader: reader, config: config, scope: scope + "/module", submodule: module6)
        module.submodule = module6
    }
}

// TODO: Convert this to suitable protocol with Discriminator configuration
extension NetD: InitializableFromPythonCheckpoint2 {
    init(reader: CheckpointReader, config: NetGConfig, scope: String) {
        let conv1 = Conv2D<Float>(reader: reader, config: config, scope: scope + "/conv1")
        let fn1 = Function<Tensor<Float>, Tensor<Float>> { leakyRelu($0) }
        let conv2 = Conv2D<Float>(reader: reader, config: config, scope: scope + "/conv2")
        let bn1 = BatchNorm<Float>(featureCount: 2 * config.lastConvFilters)
        let fn2 = Function<Tensor<Float>, Tensor<Float>> { leakyRelu($0) }
        let conv3 = Conv2D<Float>(reader: reader, config: config, scope: scope + "/conv3")
        let bn2 = BatchNorm<Float>(featureCount: 4 * config.lastConvFilters)
        let fn3 = Function<Tensor<Float>, Tensor<Float>> { leakyRelu($0) }
        let module = Sequential {
            conv1
            fn1
            conv2
            bn1
            fn2
            conv3
            bn2
            fn3
        }
          
        let module2 = Sequential {
            module
            ConvLayer(reader: reader, config: config, scope: scope + "convLayer1")
            BatchNorm<Float>(reader: reader, config: config, scope: scope)
            Function<Tensor<Float>, Tensor<Float>> { leakyRelu($0) }
            ConvLayer(reader: reader, config: config, scope: scope + "convLayer2")
        }
        
        self.module = module2
    }
}


 

