// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

// Workaround https://bugs.swift.org/browse/TF-1122 that prevents us from registering a
// loss function inside our TrainingLoop struct
public final class LossFunctionWrapper<Output: Differentiable, Target> {
  public typealias F = @differentiable (Output, @noDerivative Target) -> Tensor<Float>
  public var f: F
  init(_ f: @escaping F) { self.f = f }
}

/// Types whose elements represent a training loop.
///
/// - Note: This protocol is mainly there to give us an easy type for a generic `TrainingLoop`
///   and unless you need to rewrite your own training loop entirely, you should use `TrainingLoop`.
public protocol TrainingLoopProtocol {
  // Associatedtypes
  /// The type of the sequence of epochs for the training data.
  associatedtype Training where Training: Sequence, Training.Element: Collection,
    Training.Element.Element == LabeledData<Opt.Model.Input, Target>
  /// The type of the collection of batches for the validation data.
  associatedtype Validation where Validation: Collection,
    Validation.Element == LabeledData<Opt.Model.Input, Target>
  /// The type of the target of our model.
  associatedtype Target
  /// The type of the optimizer used.
  associatedtype Opt: Optimizer where Opt.Model: Module

  // Typealiases
  /// The type of the model.
  typealias Model = Opt.Model
  /// The type of the input of the model.
  typealias Input = Opt.Model.Input
  /// The type of the output of the model.
  typealias Output = Opt.Model.Output
  /// The type of a batch.
  typealias Batch = LabeledData<Input, Target>
  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  typealias LossFunction = LossFunctionWrapper<Output, Target>

  // Data
  /// The training epochs.
  var training: Training { get }
  /// The validation batches.
  var validation: Validation { get }

  // Model, optimizer and loss function
  /// The model.
  var model: Model { get set }
  /// The optimizer.
  var optimizer: Opt { get set }
  /// The loss function.
  var lossFunction: LossFunction { get set }

  // Callbacks
  /// The callbacks used to customize the training loop.
  var callbacks: [TrainingLoopCallback<Self>] { get set }

  // Temporary data
  /// The last input fed to the model.
  var lastInput: Input? { get set }
  /// The last target.
  var lastTarget: Target? { get set }
  /// The last predictions of the model.
  var lastOutput: Output? { get set }
  /// The last gradients computed.
  var lastGradient: Model.TangentVector? { get set }
  /// The last loss.
  var lastLoss: Tensor<Float>? { get set }
  /// The number of epochs we are currently fitting for.
  var epochCount: Int? { get set }
  /// The index of the current epoch.
  var epochIndex: Int? { get set }
  /// The number of batches in the current collection of batches.
  var batchCount: Int? { get set }
  /// The index of the current batch.
  var batchIndex: Int? { get set }
}

/// The events that occur during a call to `fit` in the `TrainingLoop`
///
/// - Note: The method is called `fit` and not `train` because it trains the model and validates it.
///   Each epoch is composed of a *training* phase and a *validation* phase.
public enum TrainingLoopEvent {
  /// The start of a fit.
  case fitStart
  /// The end of a fit.
  case fitEnd
  /// The start of one epoch (training + validation).
  case epochStart
  /// The start of one epoch (training + validation).
  case epochEnd
  /// The start of a training phase.
  case trainingStart
  /// The end of a training phase.
  case trainingEnd
  /// The start of a validation phase.
  case validationStart
  /// The end of a validation phase.
  case validationEnd
  /// The start of a training or inference step on a batch.
  case batchStart
  /// The end of a training or inference step on a batch.
  case batchEnd
  /// At the start of the optimizer update, just after the differentiable step.
  case updateStart
  /// Just after the model prediction at inference, before computing the loss.
  case inferencePredictionEnd
}

/// Callbacks that can inject custom behavior in a training loop.
public typealias TrainingLoopCallback<L: TrainingLoopProtocol>
  = (_ loop: inout L, _ event: TrainingLoopEvent) throws -> Void

/// A generic training loop.
///
/// - Parameter `Training`: the type of the sequence of epochs for training data.
/// - Parameter `Validation`: the type of the collection of batches for validation.
/// - Parameter `Target`: the type of the target.
/// - Parameter `Opt`: the type of the optimizer used.
public struct TrainingLoop<
  Training: Sequence, Validation: Collection, Target, Opt: Optimizer
>: TrainingLoopProtocol where
  Training.Element: Collection, Training.Element.Element == LabeledData<Opt.Model.Input, Target>,
  Validation.Element == LabeledData<Opt.Model.Input, Target>, Opt.Model: Module
{
  // Typealiases
  /// The type of the model.
  public typealias Model = Opt.Model
  /// The type of the input of the model.
  public typealias Input = Opt.Model.Input
  /// The type of the output of the model.
  public typealias Output = Opt.Model.Output
  /// The type of a batch.
  public typealias Batch = LabeledData<Input, Target>
  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  public typealias LossFunction = LossFunctionWrapper<Output, Target>
      
  // Data
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation
  
  // Model, optimizer and loss function
  /// The model.
  public var model: Model
  /// The optimizer.
  public var optimizer: Opt
  /// The loss function
  public var lossFunction: LossFunction
      
  // Callbacks
  /// The callbacks used to customize the training loop.
  public var callbacks: [TrainingLoopCallback<Self>] = []
  
  // Temporary data
  /// The last input fed to the model.
  public var lastInput: Input? = nil
  /// The last target.
  public var lastTarget: Target? = nil
  /// The last predictions of the model.
  public var lastOutput: Output? = nil
  /// The last gradients computed.
  public var lastGradient: Model.TangentVector? = nil
  /// The last loss.
  public var lastLoss: Tensor<Float>? = nil
  /// The number of epochs we are currently fitting for.
  public var epochCount: Int? = nil
  /// The index of the current epoch.
  public var epochIndex: Int? = nil
  /// The number of batches in the current collection of batches.
  public var batchCount: Int? = nil
  /// The index of the current batch.
  public var batchIndex: Int? = nil
      
  /// Creates an instance from `training` and `validation` data, a `model`, an `optimizer` and a
  /// `lossFunction`.
  ///
  /// Parameter callbacks: Callbacks that the `TrainingLoop` will use in every call to fit.
  public init(training: Training, validation: Validation, model: Model, optimizer: Opt,
      lossFunction: @escaping LossFunction.F, callbacks: [TrainingLoopCallback<Self>] = []) {
    self.training = training
    self.validation = validation
    self.model = model
    self.optimizer = optimizer
    self.lossFunction = LossFunction(lossFunction)
    self.callbacks = callbacks
  }
}

public extension TrainingLoop {
  /// The default differentiable step.
  mutating func differentiableStep() throws {
    guard let data = lastInput else { return }
    guard let target = lastTarget else { return }
    (lastLoss, lastGradient) = valueWithGradient(at: model) { (model: Model) -> Tensor<Float> in
      let predictions = model(data)
      lastOutput = predictions
      return lossFunction.f(predictions, target)
    }
  }
    
  /// The step used for inference.
  mutating func inferenceStep() throws {
    guard let data = lastInput else { return }
    lastOutput = model(data)
    guard let target = lastTarget else { return }
    try handleEvent(.inferencePredictionEnd)
    lastLoss = lossFunction.f(lastOutput!, target)
  }

  /// The step used for training.
  mutating func trainingStep(differentiableStep: (inout Self) throws -> Void) throws {
    try differentiableStep(&self)
    try handleEvent(.updateStart)
    optimizer.update(&model, along: lastGradient!)
  }
}

/// Control flow of the training loop.
///
/// - Note: Each of the "end" event is called after its corresponding "cancel" action for cleanup.
public enum TrainingLoopAction: Error {
  /// Abort actions in the current training/inference step and goes to the next batch.
  case cancelBatch
  /// Abort actions in the current training phase and goes to the validation phase.
  case cancelTraining
  /// Abort actions in the current validation phase and goes to the next epoch.
  case cancelValidation
  /// Abort actions in the current epoch and goes to the next epoch.
  case cancelEpoch
  /// Abort actions in the current fit and ends fitting.
  case cancelFit
}

extension TrainingLoop {
  /// Call `event` on all callbacks.
  mutating private func handleEvent(_ event: TrainingLoopEvent) throws {
    for callback in callbacks {
      try callback(&self, event)
    }
  }
}

extension TrainingLoop {
  /// Performs `step` on each of `batches`.
  mutating private func multipleSteps<Batches: Collection>(
    on batches: Batches, step: (inout Self) throws -> Void
  ) throws where Batches.Element == Batch {
    batchCount = batches.count
    for (i, batch) in batches.enumerated() {
      batchIndex = i
      (lastInput, lastTarget) = (batch.data, batch.label)
      do {
        try handleEvent(.batchStart)
        try step(&self)
      } catch TrainingLoopAction.cancelBatch {}
      try handleEvent(.batchEnd)
    }
  }
}

public extension TrainingLoop {
  /// Fit the model for `epochs` using `callbacks` to customize the default training loop.
  ///
  /// - Parameters:
  ///   - inferenceStep: The step used during the validation phase of each epoch. The default value
  ///     uses the `inferenceStep` method of `TrainingLoop`.
  ///   - trainingStep: The step used during the training phase of each epoch. The default value
  ///     uses the `trainingStep` method of `TrainingLoop`.
  mutating func fit(
    for epochs: Int, callbacks: [TrainingLoopCallback<Self>] = [],
    differentiableStep: (inout Self) throws -> Void = { try $0.differentiableStep() }
  ) throws {
    let callbacksCount = self.callbacks.count
    self.callbacks += callbacks
    defer { self.callbacks = Array(self.callbacks.prefix(callbacksCount)) }
    epochCount = epochs
      
    do{
      try handleEvent(.fitStart)
      for (i, batches) in training.prefix(epochs).enumerated() {
        epochIndex = i
        do {
          try handleEvent(.epochStart)

          // Training phase
          do {
            Context.local.learningPhase = .training
            try handleEvent(.trainingStart)
            try multipleSteps(on: batches, step: {
              try $0.trainingStep(differentiableStep: differentiableStep) })
          } catch TrainingLoopAction.cancelTraining {}
          try handleEvent(.trainingEnd)

          // Validation phase
          do {
            Context.local.learningPhase = .inference
            try handleEvent(.validationStart)
            try multipleSteps(on: validation, step: { try $0.inferenceStep() })
          } catch TrainingLoopAction.cancelValidation {}
          try handleEvent(.validationEnd)
        } catch TrainingLoopAction.cancelEpoch {}

        try handleEvent(.epochEnd)
      }
    } catch TrainingLoopAction.cancelFit {}
    try handleEvent(.fitEnd)
  }
}
