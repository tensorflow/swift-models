import TensorFlow


public func learningRateScheduler<L: TrainingLoopProtocol>(
  schedule: DecayWithOptionalWarmupSchedule,
  biasCorrectionBeta: (Float, Float)? = nil
) -> TrainingLoopCallback<L> {

  var totalStepCount: Int = 0

  return { (loop, event) throws -> Void in
    if event != .batchStart || Context.local.learningPhase == .inference { return }

    if totalStepCount == 0 {
      totalStepCount = loop.batchCount! * loop.epochCount!
    }

    let step = loop.batchIndex! + loop.epochIndex! * loop.batchCount! + 1

    var learningRate = schedule(withEndStep: totalStepCount, forStep: step)
    if let beta = biasCorrectionBeta {
      learningRate *= sqrtf(1 - powf(beta.1, Float(step))) / (beta.0 - powf(beta.1, Float(step)))
    }

    loop.optimizer.learningRate = learningRate as! L.Opt.Scalar
  }
}
