import TensorFlow

extension Array: Layer where Element: Layer, Element.Input == Element.Output {
    public typealias Input = Element.Input
    public typealias Output = Element.Output

    @differentiable(wrt: (self, input), vjp: _vjpCallAsFunction)
    @differentiable(wrt: self, vjp: _vjpCallAsFunction_wrt_self)
    public func callAsFunction(_ input: Input) -> Output {
        var activation = input
        for layer in self {
            activation = layer(activation)
        }
        return activation
    }

    public func _vjpCallAsFunction(_ input: Input)
        -> (Output, (Output.TangentVector) -> (Array.TangentVector, Input.TangentVector))
    {
        var activation = input
        var pullbacks: [(Input.TangentVector) -> (Element.TangentVector, Input.TangentVector)] = []
        for layer in self {
            let (newActivation, newPullback) = layer.valueWithPullback(at: activation) { $0($1) }
            activation = newActivation
            pullbacks.append(newPullback)
        }
        func pullback(_ v: Input.TangentVector) -> (Array.TangentVector, Input.TangentVector) {
            var activationGradient = v
            var layerGradients: [Element.TangentVector] = []
            for pullback in pullbacks.reversed() {
                let (newLayerGradient, newActivationGradient) = pullback(activationGradient)
                activationGradient = newActivationGradient
                layerGradients.append(newLayerGradient)
            }
            return (Array.TangentVector(layerGradients.reversed()), activationGradient)
        }
        return (activation, pullback)
    }

    public func _vjpCallAsFunction_wrt_self(_ input: Input)
        -> (Output, (Output.TangentVector) -> Array.TangentVector)
    {
        var activation = input
        var pullbacks: [(Input.TangentVector) -> (Element.TangentVector, Input.TangentVector)] = []
        for layer in self {
            let (newActivation, newPullback) = layer.valueWithPullback(at: activation) { $0($1) }
            activation = newActivation
            pullbacks.append(newPullback)
        }
        func pullback(_ v: Input.TangentVector) -> (Array.TangentVector) {
            var activationGradient = v
            var layerGradients: [Element.TangentVector] = []
            for pullback in pullbacks.reversed() {
                let (newLayerGradient, newActivationGradient) = pullback(activationGradient)
                activationGradient = newActivationGradient
                layerGradients.append(newLayerGradient)
            }
            return Array.TangentVector(layerGradients.reversed())
        }
        return (activation, pullback)
    }
}
