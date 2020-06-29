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

struct WordSegSettings: Codable {

  /// Dataset settings.

  /// Path to training data.
  let trainingPath: String?

  /// Path to validation data.
  let validationPath: String?

  /// Path to test data.
  let testPath: String?

  /// Model settings.

  /// Hidden unit size.
  let hiddenSize: Int

  /// Applicable to training.

  /// Dropout rate.
  let dropoutProbability: Double

  /// Power of the length penalty.
  let order: Int

  /// Weight of the length penalty.
  let lambd: Float

  /// Maximum number of training epochs.
  let maxEpochs: Int

  /// Initial learning rate.
  let learningRate: Float

  /// Backend to use.
  let backend: Backend

  /// Lexicon settings.

  /// Maximum length of a word.
  let maxLength: Int

  /// Minimum frequency of a word.
  let minFrequency: Int
}

/// Backend used to dispatch tensor operations.
enum Backend: String, Codable {
  case eager = "eager"
  case x10 = "x10"
}
