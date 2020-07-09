import TensorFlow

public func input(shape: [Int]) -> FunctionalLayer {
    return InputFunctionalLayer(shape: shape)
}

public func dense(_ prev: FunctionalLayer, outputSize: Int, activation: @escaping Dense<Float>.Activation = identity) -> FunctionalLayer {
    return FunctionalLayerWrapper(
        parent: prev,
        layer: Dense<Float>(inputSize: prev.outputShape()[0], outputSize: outputSize, activation: activation),
        outputShape: [outputSize]
    )
}

public func flatten(_ prev: FunctionalLayer) -> FunctionalLayer {
    return FunctionalLayerWrapper(
        parent: prev,
        layer: Flatten<Float>(),
        outputShape: [prev.outputShape().reduce(1, { $0 * $1 })]
    )
}

public func conv2D(
    _ prev: FunctionalLayer,
    filterShape: (Int, Int),
    outputChannels: Int,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1),
    activation: @escaping Conv2D<Float>.Activation = identity,
    useBias: Bool = true,
    filterInitializer: @escaping ParameterInitializer<Float> = glorotUniform(),
    biasInitializer: @escaping ParameterInitializer<Float> = zeros()
) -> FunctionalLayer {
    let inputShape = prev.outputShape()

    let outputShape: [Int]
    if (padding == .valid) {
        outputShape = [
            Int(ceil(Float(inputShape[0] - filterShape.0 + 1) / Float(strides.0))),
            Int(ceil(Float(inputShape[1] - filterShape.1 + 1) / Float(strides.1))),
            outputChannels
        ]
    } else {
        outputShape = [
            Int(ceil(Float(inputShape[0]) / Float(strides.0))),
            Int(ceil(Float(inputShape[1]) / Float(strides.1))),
            outputChannels
        ]
    }

    return FunctionalLayerWrapper(
        parent: prev,
        layer: Conv2D<Float>(
            filterShape: (filterShape.0, filterShape.1, inputShape[2], outputChannels),
            strides: strides, padding: padding, dilations: dilations,
            activation: activation, useBias: useBias,
            filterInitializer: filterInitializer, biasInitializer: biasInitializer
        ),
        outputShape: outputShape
    )
}

public func avgPool2D(
    _ prev: FunctionalLayer,
    poolSize: (Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid
) -> FunctionalLayer {
    let inputShape = prev.outputShape()

    let outputShape: [Int]
    if (padding == .valid) {
        outputShape = [
            Int(ceil(Float(inputShape[0] - poolSize.0 + 1) / Float(strides.0))),
            Int(ceil(Float(inputShape[1] - poolSize.1 + 1) / Float(strides.1))),
            inputShape[2]
        ]
    } else {
        outputShape = [
            Int(ceil(Float(inputShape[0]) / Float(strides.0))),
            Int(ceil(Float(inputShape[1]) / Float(strides.1))),
            inputShape[2]
        ]
    }

    return FunctionalLayerWrapper(
        parent: prev,
        layer: AvgPool2D<Float>(
            poolSize: poolSize,
            strides: strides,
            padding: padding
        ),
        outputShape: outputShape
    )
}

extension FunctionalLayer {
    public func dense(outputSize: Int, activation: @escaping Dense<Float>.Activation = identity) -> FunctionalLayer {
        return LayerInit.dense(self, outputSize: outputSize, activation: activation)
    }

    public func flatten() -> FunctionalLayer {
        return LayerInit.flatten(self)
    }

    public func conv2D(
        filterShape: (Int, Int),
        outputChannels: Int,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Conv2D<Float>.Activation = identity,
        useBias: Bool = true,
        filterInitializer: @escaping ParameterInitializer<Float> = glorotUniform(),
        biasInitializer: @escaping ParameterInitializer<Float> = zeros()
    ) -> FunctionalLayer {
        return LayerInit.conv2D(
            self,
            filterShape: filterShape,
            outputChannels: outputChannels,
            strides: strides,
            padding: padding,
            dilations: dilations,
            activation: activation,
            useBias: useBias,
            filterInitializer: filterInitializer,
            biasInitializer: biasInitializer
        )
    }

    public func avgPool2D(
        poolSize: (Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) -> FunctionalLayer {
        return LayerInit.avgPool2D(
            self,
            poolSize: poolSize,
            strides: strides,
            padding: padding
        )
    }
}
