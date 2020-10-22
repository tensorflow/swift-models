public protocol WarmupSchedule {
  var steps: Int { get set }
  var endLearningRate: Float { get set }
  func callAsFunction(forStep step: Int) -> Float
}

public protocol DecayWithOptionalWarmupSchedule {
  var warmupSchedule: WarmupSchedule? { get set }
  var startStep: Int { get set }
  var startLearningRate: Float { get set }
  var endLearningRate: Float { get set }
  init(warmupSchedule: WarmupSchedule?, startStep: Int, startLearningRate: Float, endLearningRate: Float)
  func callAsFunction(withEndStep endStep: Int, forStep step: Int) -> Float
}

extension DecayWithOptionalWarmupSchedule {
  public init(
    warmupSchedule: WarmupSchedule? = nil,
    startLearningRate: Float? = nil,
    endLearningRate: Float = 0
  ) {
    if let warmupSchedule = warmupSchedule {
      precondition(
        startLearningRate == nil,
        "Shouldn't specify startLearningRate when warmupSchedule is provided.")
      self.init(warmupSchedule: warmupSchedule, startStep: warmupSchedule.steps, startLearningRate: warmupSchedule.endLearningRate, endLearningRate: endLearningRate)
    } else {
      precondition(
        startLearningRate != nil,
        "Should specify startLearningRate when warmupSchedule is nil.")
      self.init(warmupSchedule: nil, startStep: 0, startLearningRate: startLearningRate!, endLearningRate: endLearningRate)
    }
  }
}

public struct LinearWarmupSchedule: WarmupSchedule {
  public var steps: Int
  public var endLearningRate: Float

  public init(steps: Int, endLearningRate: Float) {
    self.steps = steps
    self.endLearningRate = endLearningRate
  }

  public func callAsFunction(forStep step: Int) -> Float {
    return LinearLearningRateCurve(
      startStep: 0,
      endStep: steps,
      startLearningRate: 0,
      endLearningRate: endLearningRate)(atStep: step)
  }
}

public struct LinearDecayWithOptionalWarmupSchedule: DecayWithOptionalWarmupSchedule {
  public var warmupSchedule: WarmupSchedule? = nil
  public var startStep: Int
  public var startLearningRate: Float
  public var endLearningRate: Float

  public init(warmupSchedule: WarmupSchedule?, startStep: Int, startLearningRate: Float, endLearningRate: Float) {
    self.warmupSchedule = warmupSchedule
    self.startStep = startStep
    self.startLearningRate = startLearningRate
    self.endLearningRate = endLearningRate
  }

  public func callAsFunction(withEndStep endStep: Int, forStep step: Int) -> Float {
    if let warmupSchedule = warmupSchedule {
      if step <= warmupSchedule.steps {
        return warmupSchedule(forStep: step)
      }
    }
    return LinearLearningRateCurve(
      startStep: startStep,
      endStep: endStep,
      startLearningRate: startLearningRate,
      endLearningRate: endLearningRate)(atStep: step)

  }
}

