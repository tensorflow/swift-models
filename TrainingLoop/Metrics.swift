import TensorFlow

/// Metrics that can be registered into TrainingLoop.
public enum TrainingMetrics {
  case loss
  case accuracy

  public var name: String {
  	switch self {
  	case .loss:
  		return "loss"
  	case .accuracy:
  		return "accuracy"
  	}
  }

  public var measurer: MetricsMeasurer {
  	switch self {
  	case .loss:
  		return LossMeasurer(self.name)
  	case .accuracy:
  		return AccuracyMeasurer(self.name)
  	}
  }
}

/// A protocal defining functionalities of a metrics measurer.
public protocol MetricsMeasurer {
	var name: String { get set }
	mutating func reset()
	mutating func accumulate<Output, Target>(loss: Tensor<Float>?, predictions: Output?, labels: Target?)
	func measure() -> Float
}

/// A measurer for measuring loss.
public struct LossMeasurer: MetricsMeasurer {
	public var name: String

	private var totalBatchLoss: Float = 0
  	private var batchCount: Int32 = 0

	public init(_ name: String = "loss") {
		self.name = name
	}

	public mutating func reset() {
		totalBatchLoss = 0
		batchCount = 0
	}

	public mutating func accumulate<Output, Target>(loss: Tensor<Float>?, predictions: Output?, labels: Target?) {
		if let newBatchLoss = loss {
			totalBatchLoss += newBatchLoss.scalarized()
    		batchCount += 1
		}
	}

	public func measure() -> Float {
		return totalBatchLoss / Float(batchCount)
	}
}

/// A measurer for measuring accuracy
public struct AccuracyMeasurer: MetricsMeasurer {
	public var name: String

	private var correctGuessCount: Int32 = 0
	private var totalGuessCount: Int32 = 0

	public init(_ name: String = "accuracy") {
		self.name = name
	}

	public mutating func reset() {
		correctGuessCount = 0
		totalGuessCount = 0
	}

	public mutating func accumulate<Output, Target>(loss: Tensor<Float>?, predictions: Output?, labels: Target?) {
		guard let predictions = predictions as? Tensor<Float>, let labels = labels as? Tensor<Int32> else {
	      fatalError(
	      	"For accuracy measurements, the model output must be Tensor<Float>, and the labels must be Tensor<Int>.")
	    }
		correctGuessCount += Tensor<Int32>(predictions.argmax(squeezingAxis: 1) .== labels).sum().scalarized()
		totalGuessCount += Int32(labels.shape[0])
	}

	public func measure() -> Float {
		return Float(correctGuessCount) / Float(totalGuessCount)
	}
}
