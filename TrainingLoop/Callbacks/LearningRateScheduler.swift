import TensorFlow

/// A handler for scheduling learningRate during training.
public class LearningRateScheduler<SP: ScheduledParameter> {
  /// A function type that returns a ScheduledParameter with given `totalStepCount`.
  public typealias ScheduledParameterGetter<SP> = (
    _ totalStepCount: Float
  ) -> SP

  /// A function that returns a ScheduledParameter.
  public var scheduledParameterGetter: ScheduledParameterGetter<SP>

  public var biasCorrectionBeta1: Float?
  public var biasCorrectionBeta2: Float?

  /// Creates an instance that handles learningRate scheduling.
  /// 
  /// - Parameter scheduledParameterGetter: a function that returns a ScheduledParameter
  ///   instance which will help scheduling learning rate with particular algorithm. 
  /// - Parameter biasCorrectionBeta1: an optional coefficient for doing biasCorrection 
  ///   on the learningRate.
  /// - Parameter biasCorrectionBeta2: an optional coefficient for doing biasCorrection 
  ///   on the learningRate.
  public init(
    scheduledParameterGetter: @escaping ScheduledParameterGetter<SP>,
    biasCorrectionBeta1: Float? = nil,
    biasCorrectionBeta2: Float? = nil
  ) {
    self.scheduledParameterGetter = scheduledParameterGetter
    self.biasCorrectionBeta1 = biasCorrectionBeta1
    self.biasCorrectionBeta2 = biasCorrectionBeta2
  }

  /// A callback that will change the learning rate for the given `loop`
  /// in response of the `event`.
  ///
  /// Note: Learning rate is changed when a training step starts, and it's computed
  /// from the ScheduledParameter that self.scheduledParameterGetter returns, applied
  /// with biasCorrection if needed.
  public func schedule<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    if event != .batchStart || Context.local.learningPhase == .inference { return }

    let step = Float(loop.batchIndex! + loop.epochIndex! * loop.batchCount! + 1)
    let totalStepCount = Float(loop.batchCount! * loop.epochCount!)

    var scheduledLearningRate = scheduledParameterGetter(totalStepCount)(forStep: UInt64(step))
    if let beta1 = biasCorrectionBeta1, let beta2 = biasCorrectionBeta2 {
      scheduledLearningRate *= sqrtf(1 - powf(beta2, step)) / (1 - powf(beta1, step)) as! SP.Scalar
    }

    loop.optimizer.learningRate = scheduledLearningRate as! L.Opt.Scalar
  }
}

/// Scheduled parameter that takes the current training step as input and returns the parameter
/// value to be used for training. This will be used for scheduling the learning rate parameter.
public protocol ScheduledParameter {
  associatedtype Scalar: FloatingPoint

  /// Returns the parameter value for the specified training step.
  ///
  /// - Parameter step: Training step.
  func callAsFunction(forStep step: UInt64) -> Scalar
}

/// Dummy parameter schedule that represents no schedule being used. This is useful as a
/// default value whenever a parameter schedule argument is used.
public struct FixedParameter<Scalar: FloatingPoint>: ScheduledParameter {
  public let value: Scalar

  @inlinable
  public init(_ value: Scalar) {
    self.value = value
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    value
  }
}

extension FixedParameter: ExpressibleByFloatLiteral
where Scalar: _ExpressibleByBuiltinFloatLiteral {
  public typealias FloatLiteralType = Scalar

  public init(floatLiteral value: Scalar) {
    self.init(value)
  }
}

/// Linearly decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```swift
/// let initial = baseParameter(forStep: step)
/// let decayed = initial + step * slope
/// let decayedParameter = max(lowerBound * initial, decayed)
/// ```
public struct LinearlyDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let slope: Scalar
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new linearly decayed parameter.
  ///
  /// - Parameters:
  ///   - baseParameter: Parameter to decay.
  ///   - slope: Slope of the linear decay.
  ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
  ///   - startStep: Step after which to start decaying the parameter.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    slope: Scalar,
    lowerBound: Scalar = Scalar(0),
    startStep: UInt64 = 0
  ) {
    self.baseParameter = baseParameter
    self.slope = slope
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step < startStep { return parameter }
    let step = step - startStep
    let decayed = parameter + Scalar(step) * slope
    return max(lowerBound * parameter, decayed)
  }
}

/// Exponentially decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```swift
/// let initial = baseParameter(forStep: step)
/// let decay = decayRate ^ (step / decayStepCount)
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
/// where if `staircase = true`, then `step / decayStepCount` uses integer division and the decayed
/// parameter value follows a staircase function.
public struct ExponentiallyDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let decayRate: Scalar
  public let decayStepCount: UInt64
  public let staircase: Bool
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new exponentially decayed parameter.
  ///
  /// - Parameters:
  ///   - baseParameter: Parameter to decay.
  ///   - decayRate: Decay rate.
  ///   - decayStepCount: Decay step count.
  ///   - staircase: If `true`, the decay will occur at discrete intervals.
  ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
  ///   - startStep: Step after which to start decaying the parameter.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    decayRate: Scalar,
    decayStepCount: UInt64,
    staircase: Bool = false,
    lowerBound: Scalar = Scalar(0),
    startStep: UInt64 = 0
  ) {
    self.baseParameter = baseParameter
    self.decayRate = decayRate
    self.decayStepCount = decayStepCount
    self.staircase = staircase
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step < startStep { return parameter }
    let step = step - startStep
    let power = Scalar(step) / Scalar(decayStepCount)
    let decay = Scalar.pow(decayRate, staircase ? power.rounded(.down) : power)
    return parameter * ((1 - lowerBound) * decay + lowerBound)
  }
}

/// Reciprocal square root decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```swift
/// let initial = baseParameter(forStep: step)
/// let decay = decayFactor / sqrt(max(step, decayThreshold))
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct RSqrtDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let decayFactor: Scalar
  public let decayThreshold: Scalar
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new reciprocal square root decayed parameter.
  ///
  /// - Parameters:
  ///   - baseParameter: Parameter to decay.
  ///   - decayFactor: Decay factor.
  ///   - decayThreshold: Decay threshold.
  ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
  ///   - startStep: Step after which to start decaying the parameter.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    decayFactor: Scalar,
    decayThreshold: Scalar,
    lowerBound: Scalar = Scalar(0),
    startStep: UInt64 = 0
  ) {
    self.baseParameter = baseParameter
    self.decayFactor = decayFactor
    self.decayThreshold = decayThreshold
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step < startStep { return parameter }
    let step = step - startStep
    let decay = decayFactor / Scalar.sqrt(max(Scalar(step), decayThreshold))
    return parameter * ((1 - lowerBound) * decay + lowerBound)
  }
}

/// Cosine decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```swift
/// let initial = baseParameter(forStep: step)
/// let decay = 0.5 * (1 + cos(pi * min(step, cycleStepCount) / cycleStepCount))
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct CosineDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let cycleStepCount: UInt64
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new cosine decayed parameter.
  ///
  /// - Parameters:
  ///   - baseParameter: Parameter to decay.
  ///   - cycleStepCount: Cosine decay cycle in terms of number of steps.
  ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
  ///   - startStep: Step after which to start decaying the parameter.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    cycleStepCount: UInt64,
    lowerBound: Scalar = Scalar(0),
    startStep: UInt64 = 0
  ) {
    self.baseParameter = baseParameter
    self.cycleStepCount = cycleStepCount
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step < startStep { return parameter }
    let step = step - startStep
    let cosine = Scalar.cos(Scalar(min(step, cycleStepCount)))
    let decay = (1 + cosine) * Scalar.pi / Scalar(2 * cycleStepCount)
    return parameter * ((1 - lowerBound) * decay + lowerBound)
  }
}

/// Cycle-linear 10x decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```swift
/// let initial = baseParameter(forStep: step)
/// let cyclePosition = 1 - abs((step % (2 * cycleStepCount) - cycleStepCount) / cycleStepCount)
/// let decay = (0.1 + cyclePosition) * 3
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct CycleLinear10xDecayedParameter<
  BaseParameter: ScheduledParameter
>: ScheduledParameter {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let cycleStepCount: UInt64
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new cycle-linear 10x decayed parameter.
  ///
  /// - Parameters:
  ///   - baseParameter: Learning rate to decay.
  ///   - cycleStepCount: Cycle-linear 10x decay cycle in terms of number of steps.
  ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
  ///   - startStep: Step after which to start decaying the parameter.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    cycleStepCount: UInt64,
    lowerBound: Scalar = Scalar(0),
    startStep: UInt64 = 0
  ) {
    self.baseParameter = baseParameter
    self.cycleStepCount = cycleStepCount
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step < startStep { return parameter }
    let step = step - startStep
    let ratio = Scalar((step % (2 * cycleStepCount) - cycleStepCount)) / Scalar(cycleStepCount)
    let cyclePosition = 1 - abs(ratio)
    let decay = (1 / Scalar(10) + cyclePosition) * 3  // 10x difference in each cycle (0.3 - 3).
    return parameter * ((1 - lowerBound) * decay + lowerBound)
  }
}

/// Linearly warmed-up parameter.
///
/// For the first `warmUpStepCount` steps the base parameter is multiplied with:
/// ```
/// warmUpOffset + ((1 - warmUpOffset) / warmUpStepCount) * step
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
public struct LinearlyWarmedUpParameter<BaseParameter: ScheduledParameter>: ScheduledParameter {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let warmUpStepCount: UInt64
  public let warmUpOffset: Scalar

  /// Creates a new linear parameter warm-up schedule.
  ///
  /// - Parameters:
  ///   - baseParameter: Parameter to warm-up.
  ///   - warmUpStepCount: Number of warm-up steps.
  ///   - warmUpOffset: Linear schedule offset.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    warmUpStepCount: UInt64,
    warmUpOffset: Scalar
  ) {
    self.baseParameter = baseParameter
    self.warmUpStepCount = warmUpStepCount
    self.warmUpOffset = warmUpOffset
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step >= warmUpStepCount { return parameter }
    let factor = warmUpOffset + ((1 - warmUpOffset) / Scalar(warmUpStepCount)) * Scalar(step)
    return parameter * factor
  }
}

/// Exponentially warmed-up parameter.
///
/// For the first `warmUpStepCount` steps the base parameter is multiplied with:
/// ```
/// exp(log(warmUpFactor) / step) ^ (warmUpStepCount - step)
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
public struct ExponentiallyWarmedUpParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
  public typealias Scalar = BaseParameter.Scalar

  public let baseParameter: BaseParameter
  public let warmUpStepCount: UInt64
  public let warmUpFactor: Scalar

  /// Creates a new exponential parameter warm-up schedule.
  ///
  /// - Parameters:
  ///   - baseParameter: Parameter to warm-up.
  ///   - warmUpStepCount: Number of warm-up steps.
  ///   - warmUpFactor: Warm-up parameter scaling factor.
  @inlinable
  public init(
    baseParameter: BaseParameter,
    warmUpStepCount: UInt64,
    warmUpFactor: Scalar
  ) {
    self.baseParameter = baseParameter
    self.warmUpStepCount = warmUpStepCount
    self.warmUpFactor = warmUpFactor
  }

  @inlinable
  public func callAsFunction(forStep step: UInt64) -> Scalar {
    let parameter = baseParameter(forStep: step)
    if step >= warmUpStepCount { return parameter }
    let base = Scalar.exp(Scalar.log(warmUpFactor) / Scalar(warmUpStepCount))
    let factor = Scalar.pow(base, Scalar(warmUpStepCount - step))
    return parameter * factor
  }
}
