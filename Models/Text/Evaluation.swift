// Adapted from: https://github.com/eaplatanios/nca/blob/master/Sources/NCA/Evaluation.swift

/// Computes the Matthews correlation coefficient.
///
/// The Matthews correlation coefficient is more informative than other confusion matrix measures
/// (such as F1 score and accuracy) in evaluating binary classification problems, because it takes
/// into account the balance ratios of the four confusion matrix categories (true positives, true
/// negatives, false positives, false negatives).
///
/// - Source: [https://en.wikipedia.org/wiki/Matthews_correlation_coefficient](
///             https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).
public func matthewsCorrelationCoefficient(predictions: [Bool], groundTruth: [Bool]) -> Float {
  var tp = 0 // True positives.
  var tn = 0 // True negatives.
  var fp = 0 // False positives.
  var fn = 0 // False negatives.
  for (prediction, truth) in zip(predictions, groundTruth) {
    switch (prediction, truth) {
    case (false, false): tn += 1
    case (false, true): fn += 1
    case (true, false): fp += 1
    case (true, true): tp += 1
    }
  }
  // NOTE: Consider removing these debug print statements later.
  print("Total predictions: \(predictions.count)")
  print("True positives: \(tp)")
  print("True negatives: \(tn)")
  print("False positives: \(fp)")
  print("False negatives: \(tn)")
  let nominator = Float(tp * tn - fp * fn)
  let denominator = Float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).squareRoot()
  return denominator != 0 ? nominator / denominator : 0
}
