import TensorFlow

struct MixtureGaussians2DAndPen {
    struct Input {
        var h: Tensor<Float>
        var bias: Tensor<Float> = Tensor(0.0)
    }
    
    struct Parameters {
        var pi: Tensor<Float>
        var meanX: Tensor<Float>
        var meanY: Tensor<Float>
        var stdX: Tensor<Float>
        var stdY: Tensor<Float>
        var rho: Tensor<Float>
        var bernoulli: Tensor<Float>
    }
    
    let eps: Tensor<Float> = Tensor(1e-6)
    let mixtureComponents: Int
    
    var linear: Dense<Float>
    
    init(inputs: Int, mixtureComponents: Int) {
        self.mixtureComponents = mixtureComponents
        self.linear = Dense(inputSize: inputs, outputSize: mixtureComponents * 6 + 1)
    }
    
    func computeParameters(h: Tensor<Float>, bias: Tensor<Float>) -> Parameters {
        let yHat = linear(h)
        let M = self.mixtureComponents
        var pi = yHat.slice(lowerBounds: [0, 0], upperBounds: [yHat.shape[0], M])
        pi = pi * (1 + bias)
        pi = pi - pi.max()
        pi = softmax(pi, alongAxis: 1)
        let meanX = yHat.slice(lowerBounds: [0, M], upperBounds: [yHat.shape[0], M * 2])
        let meanY = yHat.slice(lowerBounds: [0, M * 2], upperBounds: [yHat.shape[0], M * 3])
        let stdX = exp(yHat.slice(lowerBounds: [0, M * 3],
                                  upperBounds: [yHat.shape[0], M * 4]) - bias) + eps
        let stdY = exp(yHat.slice(lowerBounds: [0, M * 4],
                                  upperBounds: [yHat.shape[0], M * 5]) - bias) + eps
        var rho = tanh(yHat.slice(lowerBounds: [0, M * 5], upperBounds: [yHat.shape[0], M * 6]))
        rho = rho / (1 + eps)
        var bernoulli = sigmoid(yHat.slice(lowerBounds: [0, M * 6],
                                           upperBounds: [yHat.shape[0], M * 6 + 1]))
        bernoulli = (bernoulli + eps) / (1 + 2 * eps)
        return .init(pi: pi, meanX: meanX, meanY: meanY,
                     stdX: stdX, stdY: stdY, rho: rho, bernoulli: bernoulli)
    }
    
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        let parameters = computeParameters(h: input.h, bias: input.bias)
        
        let mode = multinomial(parameters.pi.flattened().scalars)
        
        let mx = parameters.meanX.transposed()[mode]
        let my = parameters.meanY.transposed()[mode]
        let sx = parameters.stdX.transposed()[mode]
        let sy = parameters.stdY.transposed()[mode]
        let r = parameters.rho.transposed()[mode]
        
        let x = Tensor<Float>(randomNormal: [input.h.shape[0]])
        let y = Tensor<Float>(randomNormal: [input.h.shape[0]])
        
        let xn = (mx + sx * x).reshaped(to: [-1, 1])
        let p1 = pow(Tensor(1) - pow(r, 2), 0.5)
        let p2 = (x * r + y * p1)
        let yn = (my + sy * p2).reshaped(to: [-1, 1])
        
        let uniform = Tensor<Float>(randomUniform: [input.h.shape[0]]).scalars
        let pen = Tensor<Float>(parameters.bernoulli.flattened().scalars.enumerated()
            .map { x -> Float in x.element > uniform[x.offset] ? 1 : 0 }).reshaped(to: [-1, 1])
        
        return xn.concatenated(with: yn, alongAxis: 1).concatenated(with: pen, alongAxis: 1)
    }
}

