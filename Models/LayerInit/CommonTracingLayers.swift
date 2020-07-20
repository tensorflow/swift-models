import TensorFlow

public func input(shape: [Int]) -> TracingLayer {
    return InputTracingLayer(shape: shape)
}

public func dense(
    _ prev: TracingLayer,
    outputSize: Int,
    activation: @escaping Dense<Float>.Activation = identity
) -> TracingLayer {
    return TracingLayerWrapper(
        parent: prev,
        layer: Dense<Float>(inputSize: prev.outputShape()[0], outputSize: outputSize, activation: activation),
        outputShape: [outputSize]
    )
}

public func flatten(_ prev: TracingLayer) -> TracingLayer {
    return TracingLayerWrapper(
        parent: prev,
        layer: Flatten<Float>(),
        outputShape: [prev.outputShape().reduce(1, { $0 * $1 })]
    )
}

public func conv2D(
    _ prev: TracingLayer,
    filterShape: (Int, Int),
    outputChannels: Int,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1),
    activation: @escaping Conv2D<Float>.Activation = identity,
    useBias: Bool = true,
    filterInitializer: @escaping ParameterInitializer<Float> = glorotUniform(),
    biasInitializer: @escaping ParameterInitializer<Float> = zeros()
) -> TracingLayer {
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

    return TracingLayerWrapper(
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
    _ prev: TracingLayer,
    poolSize: (Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid
) -> TracingLayer {
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

    return TracingLayerWrapper(
        parent: prev,
        layer: AvgPool2D<Float>(
            poolSize: poolSize,
            strides: strides,
            padding: padding
        ),
        outputShape: outputShape
    )
}

public func maxPool2D(
    _ prev: TracingLayer,
    poolSize: (Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid
) -> TracingLayer {
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

    return TracingLayerWrapper(
        parent: prev,
        layer: MaxPool2D<Float>(
            poolSize: poolSize,
            strides: strides,
            padding: padding
        ),
        outputShape: outputShape
    )
}

public func globalAvgPool2D(_ prev: TracingLayer) -> TracingLayer {
    let inputShape = prev.outputShape()
    return TracingLayerWrapper(
        parent: prev,
        layer: GlobalAvgPool2D<Float>(),
        outputShape: [inputShape[2]]
    )
}

public func batchNorm(
    _ prev: TracingLayer,
    axis: Int = -1,
    momentum: Float = 0.99,
    epsilon: Float = 0.001
) -> TracingLayer {
    let prevShape = prev.outputShape()
    let featureCount = prevShape[(prevShape.count + axis) % prevShape.count]
    return TracingLayerWrapper(
        parent: prev,
        layer: BatchNorm<Float>(featureCount: featureCount, axis: axis, momentum: momentum, epsilon: epsilon),
        outputShape: prevShape
    )
}

public func merge(
    _ prev1: TracingLayer,
    _ prev2: TracingLayer,
    mergeShapes: ([Int], [Int]) -> [Int],
    mergeValues: @escaping @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
) -> TracingLayer {
    return MergeTracingLayer(
        parent1: prev1,
        parent2: prev2,
        mergeFn: mergeValues,
        outputShape: mergeShapes(prev1.outputShape(), prev2.outputShape())
    )
}

extension TracingLayer {
    public func dense(
        outputSize: Int,
        activation: @escaping Dense<Float>.Activation = identity
    ) -> TracingLayer {
        return LayerInit.dense(self, outputSize: outputSize, activation: activation)
    }

    public func flatten() -> TracingLayer {
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
    ) -> TracingLayer {
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
    ) -> TracingLayer {
        return LayerInit.avgPool2D(
            self,
            poolSize: poolSize,
            strides: strides,
            padding: padding
        )
    }

    public func maxPool2D(
        poolSize: (Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) -> TracingLayer {
        return LayerInit.maxPool2D(
            self,
            poolSize: poolSize,
            strides: strides,
            padding: padding
        )
    }

    public func globalAvgPool2D() -> TracingLayer {
        return LayerInit.globalAvgPool2D(self)
    }

    public func batchNorm(
        axis: Int = -1,
        momentum: Float = 0.99,
        epsilon: Float = 0.001
    ) -> TracingLayer {
        return LayerInit.batchNorm(
            self,
            axis: axis,
            momentum: momentum,
            epsilon: epsilon
        )
    }

    public func relu() -> TracingLayer {
        return TracingLayerWrapper(
            parent: self,
            layer: Function(TensorFlow.relu),
            outputShape: self.outputShape()
        )
    }

    public static func +(a: TracingLayer, b: TracingLayer) -> TracingLayer {
        return merge(
            a, b,
            mergeShapes: { (shape1, shape2) in
                if (shape1.elementsEqual(shape2)) {
                    fatalError("Cannot add layers with different shapes")
                } else {
                    return shape1
                }
            },
            mergeValues: { $0 + $1 }
        )
    }
}
