import TensorFlow

struct PullbackArgs<T : TensorGroup, U : TensorGroup> : TensorGroup {
  let input: T
  let cotangent: U
}

func xlaCompiled<T : Differentiable & TensorGroup, U : Differentiable & TensorGroup>(
  _ fn: @escaping @differentiable (T) -> U) -> @differentiable (T) -> U
where T.CotangentVector : TensorGroup, U.CotangentVector : TensorGroup
{
  let xlaCompiledFn: (T) -> U = _graph(fn, useXla: true)
  let xlaCompiledPullback = _graph(
    { (pbArgs: PullbackArgs<T, U.CotangentVector>) in
      pullback(at: pbArgs.input, in: fn)(pbArgs.cotangent)
    },
    useXla: true)
  return  differentiableFunction { x in
      (value: xlaCompiledFn(x),
      pullback: {
        v in
        xlaCompiledPullback(PullbackArgs(input: x, cotangent: v))}
      )
  }
}

public extension Tensor where Scalar : TensorFlowFloatingPoint {
  @inlinable @inline(__always)
  @differentiable(
    wrt: self, vjp: _vjpMean(alongAxes:)
    where Scalar : TensorFlowFloatingPoint
  )
  func mean(alongAxes axes: Tensor<Int32>) -> Tensor {
    return Raw.mean(self, reductionIndices: axes, keepDims: true)
  }

  
  @inlinable
  func _vjpMean(alongAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
    let value = mean(alongAxes: axes)
    return (value, { [shape = shapeTensor,
            count = Raw.gather(params: shapeTensor, indices: axes).product()] in
       $0.broadcast(toShape: shape) / Tensor(count)
    })
  }

  @inlinable @inline(__always)
  @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
  func variance(alongAxes axes: Tensor<Int32>) -> Tensor {
    let mean = self.mean(alongAxes: axes)
    let squaredDiff = (self - mean).squared()
    return squaredDiff.mean(alongAxes: axes)
  }
}


@_fixed_layout
public struct XLABatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The batch dimension.
    @noDerivative public let axis: Int32
    /// The momentum for the running mean and running variance.
    @noDerivative public let momentum: Tensor<Scalar>
    /// The offset value, also known as beta.
    public var offset: Tensor<Scalar>
    /// The scale value, also known as gamma.
    public var scale: Tensor<Scalar>
    /// The variance epsilon value.
    @noDerivative public let epsilon: Tensor<Scalar>
    /// The running mean.
    @noDerivative public let runningMean: Parameter<Scalar>
    /// The running variance.
    @noDerivative public let runningVariance: Parameter<Scalar>
    /// compiled batch norm
    @noDerivative
    public let compiledBatchNormTraining: @differentiable (BatchNormInput) -> BatchNormResult
  
    /// Creates a batch normalization layer.
    ///
    /// - Parameters:
    ///   - axis: The axis that should be normalized (typically the features axis).
    ///   - momentum: The momentum for the moving average.
    ///   - offset: The offset to be added to the normalized tensor.
    ///   - scale: The scale to multiply the normalized tensor by.
    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
    ///   - runningMean: The running mean.
    ///   - runningVariance: The running variance.
    public init(
        axis: Int,
        momentum: Tensor<Scalar>,
        offset: Tensor<Scalar>,
        scale: Tensor<Scalar>,
        epsilon: Tensor<Scalar>,
        runningMean: Tensor<Scalar>,
        runningVariance: Tensor<Scalar>
    ) {
        self.axis = Int32(axis)
        self.momentum = momentum
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.runningMean = Parameter(runningMean)
        self.runningVariance = Parameter(runningVariance)
    self.compiledBatchNormTraining = xlaCompiled(
      { [axis=self.axis, momentum=self.momentum, epsilon=self.epsilon]
        (arg: BatchNormInput)  in
             XLABatchNorm.batchNormTraining(
                 Tensor<Int32>(axis), momentum, arg.offset, arg.scale,
                 epsilon, arg.runningMeanValue,
                 arg.runningVarianceValue, arg.input)})
    }

    @differentiable
    private func applyingTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        // let positiveAxis = (input.rank + axis) % input.rank
        // let mean = input.mean(alongAxes: [0, positiveAxis])
        // let variance = input.variance(alongAxes: [0, positiveAxis])
        // runningMean.value += (mean - runningMean.value) * (1 - momentum)
        // runningVariance.value += (
        //     variance - runningVariance.value) * (1 - momentum)
        // let inv = rsqrt(variance + epsilon) * scale
        // return (input - mean) * inv + offset

        let result = compiledBatchNormTraining(
            BatchNormInput(  
                scale: scale, offset: offset, runningMeanValue: runningMean.value,
                runningVarianceValue: runningVariance.value, input:  input))
        runningMean.value = result.runningMeanValue
        runningVariance.value = result.runningVarianceValue
        return result.output
    }

    public struct BatchNormInput: Differentiable & TensorGroup & AdditiveArithmetic {
        let scale: Tensor<Scalar>
        let offset: Tensor<Scalar>
        let runningMeanValue: Tensor<Scalar>
        let runningVarianceValue: Tensor<Scalar>
        let input: Tensor<Scalar>
    }

    public struct BatchNormResult: Differentiable & TensorGroup & AdditiveArithmetic {
        let runningMeanValue: Tensor<Scalar>
        let runningVarianceValue: Tensor<Scalar>
        let output: Tensor<Scalar>
    }

    private static func batchNormTraining(
        _ axis: Tensor<Int32>,
        _ momentum: Tensor<Scalar>,
        _ offset: Tensor<Scalar>,
        _ scale: Tensor<Scalar>,
        _ epsilon: Tensor<Scalar>,
        _ runningMeanValue: Tensor<Scalar>,
        _ runningVarianceValue: Tensor<Scalar>,
        _ input: Tensor<Scalar>
    ) -> BatchNormResult {
        let positiveAxis = Raw.mod((input.rankTensor + axis), input.rankTensor)
        let axes = Tensor<Int32>([Tensor<Int32>(0), positiveAxis])
        let mean = input.mean(alongAxes: axes).withoutDerivative()
        let variance = input.variance(alongAxes: axes).withoutDerivative()
        let newRunningMeanValue =
            runningMeanValue + (mean - runningMeanValue) * (1 - momentum)
        let newRunningVariance =
            runningVarianceValue + (
                 variance - runningVarianceValue) * (1 - momentum)
        let inv = rsqrt(variance + epsilon) * scale
        return BatchNormResult(
            runningMeanValue: newRunningMeanValue.withoutDerivative(),
            runningVarianceValue: newRunningVariance.withoutDerivative(),
            output: (input - mean) * inv + offset
        )
    }

    @differentiable
    private func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let inv = rsqrt(runningVariance.value + epsilon) * scale
        return (input - runningMean.value) * inv + offset
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable(vjp: _vjpApplied(to:in:))
    public func applied(to input: Tensor<Scalar>, in context: Context) -> Tensor<Scalar> {
        switch context.learningPhase {
        case .training:
            return applyingTraining(to: input)
        case .inference:
            return applyingInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>, in context: Context) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (XLABatchNorm<Scalar>.CotangentVector, Tensor<Scalar>)) {
        switch context.learningPhase {
        case .training:
            return valueWithPullback(at: input) {
                $0.applyingTraining(to: $1)
            }
        case .inference:
            return valueWithPullback(at: input) {
                $0.applyingInference(to: $1)
            }
        }
    }

    /// Creates a batch normalization layer.
    ///
    /// - Parameters:
    ///   - featureCount: The number of features.
    ///   - axis: The axis that should be normalized (typically the features axis).
    ///   - momentum: The momentum for the moving average.
    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
    public init(featureCount: Int,
                axis: Int = -1,
                momentum: Tensor<Scalar> = Tensor(0.99),
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.axis = Int32(axis)
        self.momentum = momentum
        self.scale = Tensor<Scalar>(ones: [Int32(featureCount)])
        self.offset = Tensor<Scalar>(zeros: [Int32(featureCount)])
        self.epsilon = epsilon
        self.runningMean = Parameter(Tensor(0))
        self.runningVariance = Parameter(Tensor(1))
       self.compiledBatchNormTraining = xlaCompiled(
      {(arg: BatchNormInput)  in
             XLABatchNorm.batchNormTraining(
                 Tensor<Int32>(Int32(axis)), momentum, arg.offset, arg.scale,
                 epsilon, arg.runningMeanValue,
                 arg.runningVarianceValue, arg.input)})

    }
}

struct ConvBN: Layer {
    var conv: Conv2D<Float>
    var norm: XLABatchNorm<Float>

    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm = XLABatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        return norm.applied(to: conv.applied(to: input, in: context), in: context)
    }
}

