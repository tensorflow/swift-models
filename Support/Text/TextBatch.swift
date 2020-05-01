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
public struct TextBatch: KeyPathIterable {
    /// IDs that correspond to the vocabulary used while tokenizing.
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var tokenIds: Tensor<Int32>  // TODO: !!! Mutable in order to allow for batching.

    /// IDs of the token types (e.g., sentence A and sentence B in BERT).
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var tokenTypeIds: Tensor<Int32>  // TODO: !!! Mutable in order to allow for batching.

    /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var mask: Tensor<Int32>  // TODO: !!! Mutable in order to allow for batching.

    public init(tokenIds: Tensor<Int32>, tokenTypeIds: Tensor<Int32>, mask: Tensor<Int32>) {
        self.tokenIds = tokenIds
        self.tokenTypeIds = tokenTypeIds
        self.mask = mask
    }
}

// TODO: Add documentation.
public func padAndBatch(textBatches: [TextBatch], maxLength: Int? = nil) -> TextBatch {
    if textBatches.count == 1 { return textBatches.first! }
    let maxLength = maxLength ?? textBatches.map { $0.tokenIds.shape[1] }.max()!
    let paddedBatches = textBatches.map { batch -> TextBatch in
        let paddingSize = maxLength - batch.tokenIds.shape[1]
        return TextBatch(
            tokenIds: batch.tokenIds.padded(forSizes: [
                (before: 0, after: 0),
                (before: 0, after: paddingSize),
            ]),
            tokenTypeIds: batch.tokenTypeIds.padded(forSizes: [
                (before: 0, after: 0),
                (before: 0, after: paddingSize),
            ]),
            mask: batch.mask.padded(forSizes: [
                (before: 0, after: 0),
                (before: 0, after: paddingSize),
            ]))
    }
    return TextBatch(
        tokenIds: Tensor<Int32>(
            concatenating: paddedBatches.map { $0.tokenIds }, alongAxis: 0),
        tokenTypeIds: Tensor<Int32>(
            concatenating: paddedBatches.map { $0.tokenTypeIds }, alongAxis: 0),
        mask: Tensor<Int32>(
            concatenating: paddedBatches.map { $0.mask }, alongAxis: 0))
}
