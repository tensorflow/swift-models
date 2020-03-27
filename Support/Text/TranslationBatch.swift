//
//  File.swift
//  
//
//  Created by Andre Carrera on 3/26/20.
//

import Foundation
import TensorFlow

public struct TranslationBatch: KeyPathIterable {
    /// IDs that correspond to the vocabulary used while tokenizing.
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var tokenIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    // aka src
    
    public var targetTokenIds: Tensor<Int32>
    // aka tgt
    
    /// IDs of the token types (e.g., sentence A and sentence B in BERT).
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    //    public var tokenTypeIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    
    /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var mask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetMask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetTruth: Tensor<Int32>
    
    public var tokenCount: Int32
    
    public init(source: Tensor<Int32>, target: Tensor<Int32>, sourcePadId: Int32, targetPadId: Int32) {
        self.tokenIds = source
        self.mask = Tensor<Float>(Tensor(zerosLike: source)
            .replacing(with: Tensor(onesLike: source), where: source .!= Tensor.init(sourcePadId))
            .expandingShape(at: 1))
        
        let rangeExceptLast = 0..<(target.shape[1] - 1)
        self.targetTokenIds = target[0...,rangeExceptLast]
        self.targetTruth = target[0..., 1...]
        self.targetMask = TranslationBatch.makeStandardMask(target: self.targetTokenIds, pad: targetPadId)
        self.tokenCount = Tensor(zerosLike: targetTruth)
            .replacing(with: Tensor(onesLike: targetTruth), where: self.targetTruth .!= Tensor.init(targetPadId))
            .sum().scalar!
        
    }
    
    public init(tokenIds: Tensor<Int32>, targetTokenIds: Tensor<Int32>, mask: Tensor<Float>, targetMask: Tensor<Float>, targetTruth: Tensor<Int32>, tokenCount: Int32) {
        self.tokenIds = tokenIds
        self.targetTokenIds = targetTokenIds
        self.mask = mask
        self.targetMask = targetMask
        self.targetTruth = targetTruth
        self.tokenCount = tokenCount
    }
    
    static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask(size: target.shape.last!)
        return Tensor<Float>(targetMask)
    }
}

public func subsequentMask(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}
