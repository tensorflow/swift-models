import TensorFlow

/// Returns a TrainingLoop callback that will change the learning rate according to `schedule`.
public func learningRateScheduler<L: TrainingLoopProtocol>(
  schedule: @escaping (Int) -> LearningRateSchedule,
  biasCorrectionBeta: (Float, Float)? = nil
) -> TrainingLoopCallback<L> {

  var totalStepCount: Int = 0

  return { (loop, event) throws -> Void in
    if event != .batchStart || Context.local.learningPhase == .inference { return }

    if totalStepCount == 0 {
      totalStepCount = loop.batchCount! * loop.epochCount!
    }

    let step = loop.batchIndex! + loop.epochIndex! * loop.batchCount!

    var learningRate = schedule(totalStepCount)(step)
    if let beta = biasCorrectionBeta {
      learningRate *= sqrt(1 - pow(beta.1, Float(step))) / (beta.0 - pow(beta.1, Float(step)))
    }

    loop.optimizer.learningRate = learningRate as! L.Opt.Scalar
  }
}

/// A segment of the learning rate schedule.
public struct ScheduleSegment {
  /// Shape of the segment.
  public var shape: Shape
  /// The start learning rate.
  public var startRate: Float?
  /// The end learning rate.
  public var endRate: Float
  /// Count of steps across the segment.
  public var stepCount: Int?

  /// Creates a `stepCount`-step segment with shape `shape`, start rate `startRate` and end rate `endRate`.
  public init(shape: Shape, startRate: Float? = nil, endRate: Float, stepCount: Int? = nil) {
    self.shape = shape
    self.startRate = startRate
    self.endRate = endRate
    self.stepCount = stepCount
  }
}

/// Returns a function that returns a LearningRateSchedule given totalStepCount; the function
/// is constucted from an array of `schedules`. 
public func makeSchedule(_ schedules: [ScheduleSegment]) -> (Int) -> LearningRateSchedule {
  precondition(schedules.count > 0)

  return { (totalStepCount: Int) -> LearningRateSchedule in
    var lrs = LearningRateSchedule(startRate: schedules.first!.startRate ?? 0)
  
    var lastEndStep = 0
    for (i, s) in schedules.enumerated() {
      var stepCount: Int

      if i < schedules.count - 1 {
        precondition(s.stepCount != nil)
        stepCount = s.stepCount! 
        lastEndStep += (stepCount - 1)
      } else {
        if s.stepCount == nil {
          stepCount = totalStepCount - lastEndStep
        } else {
          stepCount = s.stepCount! 
        }
      }
      lrs.appendSegment(stepCount: stepCount, shape: s.shape, startRate: s.startRate, endRate: s.endRate)
    }
    return lrs
  }
}
