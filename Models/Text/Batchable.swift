// Adapted from: https://github.com/eaplatanios/nca/blob/master/Sources/NCA/Utilities/Protocols.swift

import TensorFlow
public protocol Batchable {
  static func batch(_ values: [Self]) -> Self
}

extension Tensor: Batchable {
  public static func batch(_ values: [Tensor]) -> Tensor {
    Tensor(stacking: values, alongAxis: 0)
  }
}

extension KeyPathIterable {
  public static func batch(_ values: [Self]) -> Self {
    var result = values[0]
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<UInt8>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int32>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int64>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    return result
  }
}
