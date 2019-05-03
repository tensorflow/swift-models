import TensorFlow

extension Array: Layer where Element: Layer, Element.Input == Element.Output {
    public typealias Input = Element.Input
    public typealias Output = Element.Output

    @differentiable(vjp: _gradCall)
    public func call(_ input: Input) -> Output {
        var activation = input
        for layer in self {
            activation = layer(activation)
        }
        return activation
    }

    public func _gradCall(_ input: Input)
        -> (Output, (Output.CotangentVector) -> (Array.CotangentVector, Input.CotangentVector))
    {
        var activation = input
        var pullbacks: [(Input.CotangentVector) -> (Element.CotangentVector, Input.CotangentVector)] = []
        for layer in self {
            let (newActivation, newPullback) = layer.valueWithPullback(at: activation) { $0($1) }
            activation = newActivation
            pullbacks.append(newPullback)
        }
        func pullback(_ v: Input.CotangentVector) -> (Array.CotangentVector, Input.CotangentVector) {
            var activationGradient = v
            var layerGradients: [Element.CotangentVector] = []
            for pullback in pullbacks.reversed() {
                let (newLayerGradient, newActivationGradient) = pullback(activationGradient)
                activationGradient = newActivationGradient
                layerGradients.append(newLayerGradient)
            }
            return (Array.CotangentVector(layerGradients.reversed()), activationGradient)
        }
        return (activation, pullback)
    }
}
