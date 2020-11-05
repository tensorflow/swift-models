// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import Checkpoints
import Foundation
import ModelSupport


public struct NetG: Layer {
    
    // TODO: need to persist a sample generator and set this URL
    public static let remoteCheckpoint: URL =
        URL(string: "")!

    var module: UNetSkipConnectionOutermost<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost>>>>>>>
    public init(inputChannels: Int, outputChannels: Int, ngf: Int, useDropout: Bool = false) {
        let firstBlock = UNetSkipConnectionInnermost(inChannels: ngf * 8, innerChannels: ngf * 8, outChannels: ngf * 8)
        
        let module1 = UNetSkipConnection(inChannels: ngf * 8, innerChannels: ngf * 8, outChannels: ngf * 8,
                                         submodule: firstBlock, useDropOut: useDropout)
        let module2 = UNetSkipConnection(inChannels: ngf * 8, innerChannels: ngf * 8, outChannels: ngf * 8,
                                         submodule: module1, useDropOut: useDropout)
        let module3 = UNetSkipConnection(inChannels: ngf * 8, innerChannels: ngf * 8, outChannels: ngf * 8,
                                         submodule: module2, useDropOut: useDropout)

        let module4 = UNetSkipConnection(inChannels: ngf * 4, innerChannels: ngf * 8, outChannels: ngf * 4,
                                         submodule: module3, useDropOut: useDropout)
        let module5 = UNetSkipConnection(inChannels: ngf * 2, innerChannels: ngf * 4, outChannels: ngf * 2,
                                         submodule: module4, useDropOut: useDropout)
        let module6 = UNetSkipConnection(inChannels: ngf, innerChannels: ngf * 2, outChannels: ngf,
                                         submodule: module5, useDropOut: useDropout)

        self.module = UNetSkipConnectionOutermost(inChannels: inputChannels, innerChannels: ngf, outChannels: outputChannels,
                                                  submodule: module6)
    }

    public init(checkpoint: URL = NetG.remoteCheckpoint) throws {
        let parameters = NetGConfig(
            inChannels: 3, outChannels: 3, ngf: 64,
            useDropout: false, lastConvFilters: 64,
            learningRate: 0.0002, beta: 0.5, padding: 1,
            kernelSize: 4)

        // Try loading from the given checkpoint.
        do {
            let auxiliary: [String] = [
                  "checkpoint",
                  "hparams.json"
            ]

            let reader: CheckpointReader = try CheckpointReader(
                checkpointLocation: checkpoint,
                modelName: "NetG-\(checkpoint.pathComponents.dropLast().last ?? "")",
                additionalFiles: auxiliary)
            // TODO(michellecasbon): expose this.
            reader.isCRCVerificationEnabled = false

            // Initialize a model with the given config.
            let gen = NetG(reader: reader, config: parameters, scope: "model")
            module = gen.module
            print("generator loaded from checkpoint successfully.")
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load generator from checkpoint. \(error)")
            throw error
        }

        print("Generator init complete.")

    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return self.module(input)
    }
}


extension NetG {
    public func writeCheckpoint(to location: URL, name: String) throws {
        var tensors = [String: Tensor<Float>]()
        recursivelyObtainTensors(self, scope: "model", tensors: &tensors, separator: "/")
        let writer = CheckpointWriter(tensors: tensors)
        try writer.write(to: location, name: name)
        
        // TODO: Copy auxiliary files if they need to be in different location than current
        // local storage.
//        if location != storage {
//            try writeAuxiliary(to: location)
//        }
    }
    
//    public func writeAuxiliary(to location: URL) throws {
//        let fileSystem = FoundationFileSystem()
//        let vocabularyFileURL: URL = storage.appendingPathComponent("encoder.json")
//        let mergesFileURL: URL = storage.appendingPathComponent("vocab.bpe")
//        let hparamsFileURL: URL = storage!.appendingPathComponent("hparams.json")
////        let destinationEncoderURL: URL = location.appendingPathComponent("encoder.json")
//        let destinationMergesURL: URL = location.appendingPathComponent("vocab.bpe")
//        let destinationHparamsURL: URL = location.appendingPathComponent("hparams.json")

//        try fileSystem.copy(source: vocabularyFileURL, dest: destinationEncoderURL)
//        try fileSystem.copy(source: mergesFileURL, dest: destinationMergesURL)
//        try fileSystem.copy(source: hparamsFileURL, dest: destinationHparamsURL)
//    }
}

extension NetG: Equatable {
    public static func == (left: NetG, right: NetG) -> Bool {
        return left.module == right.module
    }
}
