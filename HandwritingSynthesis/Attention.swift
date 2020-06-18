import TensorFlow

struct Attention {
    struct Input {
        var h1: Tensor<Float>
        var kappaPrev: Tensor<Float>
        var cSeqLen: Float
    }
    
    struct Output {
        var phi: Tensor<Float>
        var kappa: Tensor<Float>
    }
    
    let mixtureComponents: Int
    var linear: Dense<Float>
    
    init(inputs: Int, mixtureComponents: Int) {
        self.mixtureComponents = mixtureComponents
        self.linear = Dense(inputSize: inputs, outputSize: mixtureComponents * 3)
    }
    
    func callAsFunction(_ input: Input) -> Output {
        let K = self.mixtureComponents
        let params = exp(linear(input.h1))
        let alpha = params.slice(lowerBounds: [0, 0], upperBounds: [params.shape[0], K])
        let beta = params.slice(lowerBounds: [0, K], upperBounds: [params.shape[0], K * 2])
        let kappa = input.kappaPrev + 0.1 * params.slice(lowerBounds: [0, K * 2],
                                                         upperBounds: [params.shape[0],
                                                                       params.shape[1]])
        let u = Tensor<Float>(rangeFrom: 0, to: input.cSeqLen, stride: 1).reshaped(to: [-1, 1, 1])
        let phi = (alpha * exp(-beta * pow((kappa - u), 2))).sum(squeezingAxes: -1)
        return .init(phi: phi, kappa: kappa)
    }
}
