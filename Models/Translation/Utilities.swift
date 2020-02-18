//
//  Utilities.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow
import Foundation
public typealias Activation<Scalar: TensorFlowFloatingPoint> =
    @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

public typealias ActivationInput<Input: Differentiable,Scalar: TensorFlowFloatingPoint> =
@differentiable (Input) -> Tensor<Scalar>

//struct SubLayerInput<Input: Differentiable,Scalar: TensorFlowFloatingPoint >: Differentiable {
//    var sequence: Input
//    @noDerivative let activation: ActivationInput<Input,Scalar>
//    @differentiable
//    init(sequence: Input, activation: @escaping ActivationInput<Input,Scalar>) {
//        self.sequence = sequence
//        self.activation = activation
//    }
//}

struct DecoderContext: Differentiable {
    var decoder: TransformerDecoderLayer,
    input: DecoderInput<Float>
    
    @differentiable
    init(decoder: TransformerDecoderLayer,
    input: DecoderInput<Float>) {
        self.decoder = decoder
        self.input = input
    }
}

struct SubLayerInput<Scalar: TensorFlowFloatingPoint >: Differentiable {
    var sequence: Tensor<Scalar>
//    var context: DecoderContext
    // if I want to use encoder context I could refactor to use an enum
    @noDerivative public let activation: SubLayerInput<Scalar>.Activation

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    @differentiable
    init(sequence: Tensor<Scalar>, activation: @escaping SubLayerInput<Scalar>.Activation) {
        self.sequence = sequence
        self.activation = activation
//        self.context = context
    }
}

// could try putting activation in sublayer connection
// that would mean I would change the value of the activation during the function call.
struct SublayerConnection: Layer {
    var norm: LayerNorm<Float>
    var dropout: Dropout<Float>
    init(size: Int, droputProb: Double) {
        self.norm =  LayerNorm(featureCount: size, axis: -2)// todo check axis
        self.dropout = Dropout(probability: droputProb)
    }
    @differentiable
    func callAsFunction(_ input: SubLayerInput< Float>) -> Tensor<Float> {
// the old call, can't use it because activation now takes a tuple.
        return input.sequence + self.dropout(input.activation(self.norm(input.sequence)))
//        return input.context.input.sequence + self.dropout(input.activation(self.norm(input.context.input.sequence), input.context))
    }
}

struct PositionwiseFeedForward: Layer {
    // "Implements FFN equation."
    var dense1: TimeDistributed// TODO might need to be TimeDistributed to handle timesteps??
    var dense2: TimeDistributed
    @noDerivative let dropout: Dropout<Float>
    
    init(dimensionalityModel:Int, innerLayerDimensionality:Int, dropProbability: Double=0.1) {
        // these are just "nn.Linear"
        dense1 = TimeDistributed(Dense(inputSize: dimensionalityModel, outputSize: innerLayerDimensionality))
        dense2 = TimeDistributed(Dense(inputSize: innerLayerDimensionality, outputSize: dimensionalityModel))
        dropout = Dropout<Float>(probability: dropProbability)
    }
    
//    @differentiable
//    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
//        return dense2(dropout(relu(dense1(input))))
//    }
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return relu(dense1(input)).sequenced(through: dense2, dropout)
//        return relu(dense1(input)).sequenced(through: dense2, dropout)
    }
}


struct TimeDistributed: Layer {
    var dense: Dense<Float>
    
    init(_ wrapped: Dense<Float>) {
        self.dense = wrapped
    }
    
    @differentiable(wrt: (self, input))
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let reshaped = input.reshaped(to: [batchSize * timeSteps, features])
        let output = dense(reshaped)
        let outputFeatures = output.shape[1]
        return output.reshaped(to: [batchSize, timeSteps, outputFeatures])
    }
}

struct PositionalEncoding: ParameterlessLayer {
    @noDerivative var encoding: Parameter<Float> // maybe should be an embedding?
    @noDerivative var dropout: Dropout<Float>
    init(size: Int, dropoutProbability: Double = 0, maxLength:Int=5000) {
        dropout = Dropout(probability: dropoutProbability)
        let position = Tensor(rangeFrom: 0, to: Float(maxLength), stride: 1).expandingShape(at: 1)
        let divStart = stride(from: 0, to: size, by: 2).map{ Float($0)}
        let divTerm = Tensor(divStart.map{ 1.0 / pow(10000.0, $0 / Float(size)) })
        
        //        var positionalEncoding = Tensor<Float>(zeros: [maxLength, size]).array
        //        let sinPortion = sin(position * divTerm).array
        //        let cosPortion = cos(position * divTerm).array
        //        // todo use ranges, just like the callAsFunction below
        //        for index in positionalEncoding.indices {
        //            let scalarsArrays = [sinPortion[index].scalars,cosPortion[index].scalars]
        //            positionalEncoding[index] = ShapedArraySlice(shape: [sinPortion[index].count + cosPortion[index].count], scalars:
        //                // alternates the two tensors
        //                (0..<scalarsArrays.map{$0.count}.max()!)
        //            .flatMap{i in scalarsArrays.filter{i<$0.count}.map{$0[i]} } )
        //        }
        
        var positionalEncoding = Tensor<Float>(zeros: [maxLength, size])
        // just alternating the value that is placed inside the tensor between to arrays.
        positionalEncoding[0..., 0..<positionalEncoding.shape[1]..2] = sin(position * divTerm)
        positionalEncoding[0..., 1..<positionalEncoding.shape[1]..2] = cos(position * divTerm)
        encoding = Parameter(positionalEncoding.expandingShape(at: 0))
        //        encoding = Parameter(Tensor(positionalEncoding).expandingShape(at: 0))
    }
    
    @differentiable(wrt: (self, input))
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return self.dropout(input + encoding.value[0..., 0..<input.shape[1]])
    }
}
public struct TransformerInput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    /// Sequence that the transformer encoder operates over. The shape of this tensor is
    /// `[batchSize, sequenceLength, depth]` or `[batchSize, sequenceLength * depth]`.
    public var sequence: Tensor<Scalar>
    
    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or
    /// `0`. The attention scores will effectively be set to negative infinity for any positions in
    /// the mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    public var attentionMask: Tensor<Scalar>
    
    /// The batch size of this input. This is optional because it is only needed if the input
    /// sequences have been reshaped to matrices.
    @noDerivative let batchSize: Int?
    
    @differentiable
    public init(sequence: Tensor<Scalar>, attentionMask: Tensor<Scalar>, batchSize: Int? = nil) {
        self.sequence = sequence
        self.attentionMask = attentionMask
        self.batchSize = batchSize
    }
}


extension KeyPathIterable {
    public mutating func clipByGlobalNorm<Scalar: TensorFlowFloatingPoint>(clipNorm: Scalar) {
        let clipNorm = Tensor<Scalar>(clipNorm)
        var globalNorm = Tensor<Scalar>(zeros: [])
        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            globalNorm += self[keyPath: kp].squared().sum()
        }
        globalNorm = sqrt(globalNorm)
        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            self[keyPath: kp] *= clipNorm / max(globalNorm, clipNorm)
        }
    }
}

extension Tensor {
    /// Returns this tensor reshaped to a matrix (i.e., a rank-2 tensor).
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedToMatrix() -> Tensor {
        reshaped(to: [-1, shape[rank - 1]])
    }

    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedFromMatrix(originalShape: TensorShape) -> Tensor {
        reshaped(to: TensorShape(
            originalShape[0..<originalShape.count - 1].dimensions + [shape[rank - 1]]))
    }

    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedFromMatrix(originalShape: Tensor<Int32>) -> Tensor {
        reshaped(toShape: Tensor<Int32>(concatenating: [
            originalShape[0..<originalShape.shape[0] - 1],
            Tensor<Int32>([Int32(shape[rank - 1])])
        ]))
    }
}

/// Downloads the file at `url` to `path`, if `path` does not exist.
///
/// - Parameters:
///   - from: URL to download data from.
///   - to: Destination file path.
///
/// - Returns: Boolean value indicating whether a download was
///     performed (as opposed to not needed).
internal func maybeDownload(from url: URL, to destination: URL) throws {
    if !FileManager.default.fileExists(atPath: destination.path) {
        // Create any potentially missing directories.
        try FileManager.default.createDirectory(
            atPath: destination.deletingLastPathComponent().path,
            withIntermediateDirectories: true)

        // Create the URL session that will be used to download the dataset.
        let semaphore = DispatchSemaphore(value: 0)
        let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)

        // Download the data to a temporary file and then copy that file to
        // the destination path.
        print("Downloading \(url).")
        let task = session.downloadTask(with: url)
        task.resume()

        // Wait for the download to finish.
        semaphore.wait()
    }
}

internal class DataDownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let destinationFileUrl: URL
    let semaphore: DispatchSemaphore
    let numBytesFrequency: Int64

    internal var logCount: Int64 = 0

    init(
        destinationFileUrl: URL,
        semaphore: DispatchSemaphore,
        numBytesFrequency: Int64 = 1024 * 1024
    ) {
        self.destinationFileUrl = destinationFileUrl
        self.semaphore = semaphore
        self.numBytesFrequency = numBytesFrequency
    }

    internal func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) -> Void {
        do {
            try FileManager.default.moveItem(at: location, to: destinationFileUrl)
        } catch (let writeError) {
            print("Error writing file \(location.path) : \(writeError)")
        }
        print("Downloaded successfully to \(location.path).")
        semaphore.signal()
    }
}

internal func extract(zipFileAt source: URL, to destination: URL) throws {
    print("Extracting file at '\(source.path)'.")
    let process = Process()
    process.environment = ProcessInfo.processInfo.environment
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", "unzip -d \(destination.path) \(source.path)"]
    try process.run()
    process.waitUntilExit()
}

internal func extract(tarGZippedFileAt source: URL, to destination: URL) throws {
    print("Extracting file at '\(source.path)'.")
    try FileManager.default.createDirectory(
        at: destination,
        withIntermediateDirectories: false)
    let process = Process()
    process.environment = ProcessInfo.processInfo.environment
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", "tar -C \(destination.path) -xzf \(source.path)"]
    try process.run()
    process.waitUntilExit()
}

internal func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
    try Data(contentsOf: fileURL).withUnsafeBytes {
        $0.split(separator: UInt8(ascii: "\n")).map {
            $0.split(separator: UInt8(ascii: "\t"), omittingEmptySubsequences: false)
                .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
        }
    }
}
