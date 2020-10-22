import TensorFlow

public protocol LearningRateCurve {
  var startStep: Int { get set }
  var endStep: Int { get set }
  var startLearningRate: Float { get set }
  var endLearningRate: Float { get set }

  func callAsFunction(atStep step: Int) -> Float
}

public struct LinearLearningRateCurve: LearningRateCurve {
  public var startStep: Int
  public var endStep: Int
  public var startLearningRate: Float
  public var endLearningRate: Float

  public func callAsFunction(atStep step: Int) -> Float {
    let slope = (endLearningRate - startLearningRate) / Float(endStep - startStep)
    return startLearningRate + Float(step - startStep) * slope 
  }
}
