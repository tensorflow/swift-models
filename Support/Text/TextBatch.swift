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

/// Tokenized text passage.
public struct TextBatch {
  /// IDs that correspond to the vocabulary used while tokenizing.
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public let tokenIds: Tensor<Int32>

  /// IDs of the token types (e.g., sentence A and sentence B in BERT).
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public let tokenTypeIds: Tensor<Int32>

  /// Mask over the sequence of tokens specifying which ones are "real" as 
  /// opposed to "padding".
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public let mask: Tensor<Int32>

  public init(
    tokenIds: Tensor<Int32>, tokenTypeIds: Tensor<Int32>, mask: Tensor<Int32>
  ) {
    self.tokenIds = tokenIds
    self.tokenTypeIds = tokenTypeIds
    self.mask = mask
  }
}

// TODO: When derived conformance to Collatable is implemented in swift-apis, 
// this won't be necessary.
extension TextBatch: Collatable {
  /// Creates an instance from collating `samples`.
  public init<BatchSamples: Collection>(collating samples: BatchSamples)
  where BatchSamples.Element == Self {
    self.init(
      tokenIds: .init(collating: samples.map(\.tokenIds)), 
      tokenTypeIds: .init(collating: samples.map(\.tokenTypeIds)), 
      mask: .init(collating: samples.map(\.mask))
    )
  }
}

extension Collection where Element == TextBatch {
  /// Returns the elements of `self`, padded to `maxLength` if specified
  /// or the maximum length of the elements in `self` otherwise.
  public func paddedAndCollated(to maxLength: Int? = nil) -> TextBatch {
    if count == 1 { return first! }
    let maxLength = maxLength ?? self.map { $0.tokenIds.shape[1] }.max()!
    let paddedTexts = self.map { text -> TextBatch in
      let paddingSize = maxLength - text.tokenIds.shape[1]
      return TextBatch(
        tokenIds: text.tokenIds.padded(forSizes: [
          (before: 0, after: 0),
          (before: 0, after: paddingSize)]),
        tokenTypeIds: text.tokenTypeIds.padded(forSizes: [
          (before: 0, after: 0),
          (before: 0, after: paddingSize)]),
        mask: text.mask.padded(forSizes: [
          (before: 0, after: 0),
          (before: 0, after: paddingSize)]))
    }
    return paddedTexts.collated
  }
}