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
import Foundation
import ModelSupport

// Original Paper:
// "Very Deep Convolutional Networks for Large-Scale Image Recognition"
// Karen Simonyan, Andrew Zisserman
// https://arxiv.org/abs/1409.1556

public struct VGGBlock: Layer {
    var blocks: [Conv2D<Float>] = []
    var maxpool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    public init(featureCounts: (Int, Int, Int, Int), blockCount: Int) {
        self.blocks = [Conv2D<Float>(filterShape: (3, 3, featureCounts.0, featureCounts.1),
            padding: .same,
            activation: relu)]
        for _ in 1..<blockCount {
            self.blocks += [Conv2D<Float>(filterShape: (3, 3, featureCounts.2, featureCounts.3),
                padding: .same,
                activation: relu)]
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return maxpool(blocks.differentiableReduce(input) { $1($0) })
    }
}

public struct VGG16: Layer {
    var layer1: VGGBlock
    var layer2: VGGBlock
    var layer3: VGGBlock
    var layer4: VGGBlock
    var layer5: VGGBlock

    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 512 * 7 * 7, outputSize: 4096, activation: relu)
    var dense2 = Dense<Float>(inputSize: 4096, outputSize: 4096, activation: relu)
    var output: Dense<Float>

    public init(classCount: Int = 1000) {
        layer1 = VGGBlock(featureCounts: (3, 64, 64, 64), blockCount: 2)
        layer2 = VGGBlock(featureCounts: (64, 128, 128, 128), blockCount: 2)
        layer3 = VGGBlock(featureCounts: (128, 256, 256, 256), blockCount: 3)
        layer4 = VGGBlock(featureCounts: (256, 512, 512, 512), blockCount: 3)
        layer5 = VGGBlock(featureCounts: (512, 512, 512, 512), blockCount: 3)
        output = Dense(inputSize: 4096, outputSize: classCount, activation: softmax)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let backbone = input.sequenced(through: layer1, layer2, layer3, layer4, layer5)
        return backbone.sequenced(through: flatten, dense1, dense2, output)
    }
}

public struct VGG19: Layer {
    var layer1: VGGBlock
    var layer2: VGGBlock
    var layer3: VGGBlock
    var layer4: VGGBlock
    var layer5: VGGBlock

    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 512 * 7 * 7, outputSize: 4096, activation: relu)
    var dense2 = Dense<Float>(inputSize: 4096, outputSize: 4096, activation: relu)
    var output: Dense<Float>

    public init(classCount: Int = 1000) {
        layer1 = VGGBlock(featureCounts: (3, 64, 64, 64), blockCount: 2)
        layer2 = VGGBlock(featureCounts: (64, 128, 128, 128), blockCount: 2)
        layer3 = VGGBlock(featureCounts: (128, 256, 256, 256), blockCount: 4)
        layer4 = VGGBlock(featureCounts: (256, 512, 512, 512), blockCount: 4)
        layer5 = VGGBlock(featureCounts: (512, 512, 512, 512), blockCount: 4)
        output = Dense(inputSize: 4096, outputSize: classCount, activation: softmax)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let backbone = input.sequenced(through: layer1, layer2, layer3, layer4, layer5)
        return backbone.sequenced(through: flatten, dense1, dense2, output)
    }
}

public func extract(tarGZippedFileAt source: URL, to destination: URL) throws {
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

extension VGG16 {
    public func pretrained(from directory:URL) throws -> VGG16 {
        print("Loading VGG16 pre-trained on Imagenet-2012.")
        var url = URL(string: "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz")!

        let subDirectory = "vgg_16_2016_08_28"
        let compressedFileURL = directory.appendingPathComponent("\(subDirectory).tar.gz")
        try download(from: url, to:directory)
        let extractedDirectoryURL = directory.appendingPathComponent(subDirectory)
        if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
            try extract(tarGZippedFileAt: compressedFileURL, to: extractedDirectoryURL)
        }
        var vgg16 = VGG16()
        let path = extractedDirectoryURL.appendingPathComponent("vgg_16.ckpt")
        let checkpointReader = TensorFlowCheckpointReader(checkpointPath: path.path)
        
        vgg16.layer1.blocks[0].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv1/conv1_1/weights"))
        vgg16.layer1.blocks[0].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv1/conv1_1/biases"))
        vgg16.layer1.blocks[1].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv1/conv1_2/weights"))
        vgg16.layer1.blocks[1].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv1/conv1_2/biases"))

        vgg16.layer2.blocks[0].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv2/conv2_1/weights"))
        vgg16.layer2.blocks[0].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv2/conv2_1/biases"))
        vgg16.layer2.blocks[1].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv2/conv2_2/weights"))
        vgg16.layer2.blocks[1].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv2/conv2_2/biases"))

        vgg16.layer3.blocks[0].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv3/conv3_1/weights"))
        vgg16.layer3.blocks[0].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv3/conv3_1/biases"))
        vgg16.layer3.blocks[1].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv3/conv3_2/weights"))
        vgg16.layer3.blocks[1].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv3/conv3_2/biases"))
        vgg16.layer3.blocks[2].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv3/conv3_3/weights"))
        vgg16.layer3.blocks[2].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv3/conv3_3/biases"))

        vgg16.layer4.blocks[0].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv4/conv4_1/weights"))
        vgg16.layer4.blocks[0].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv4/conv4_1/biases"))
        vgg16.layer4.blocks[1].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv4/conv4_2/weights"))
        vgg16.layer4.blocks[1].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv4/conv4_2/biases"))
        vgg16.layer4.blocks[2].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv4/conv4_3/weights"))
        vgg16.layer4.blocks[2].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv4/conv4_3/biases"))

        vgg16.layer5.blocks[0].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv5/conv5_1/weights"))
        vgg16.layer5.blocks[0].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv5/conv5_1/biases"))
        vgg16.layer5.blocks[1].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv5/conv5_2/weights"))
        vgg16.layer5.blocks[1].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv5/conv5_2/biases"))
        vgg16.layer5.blocks[2].filter = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv5/conv5_3/weights"))
        vgg16.layer5.blocks[2].bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/conv5/conv5_3/biases"))

        vgg16.dense1.weight = Tensor(checkpointReader.loadTensor(named: "vgg_16/fc6/weights"))
        vgg16.dense1.bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/fc6/weights"))
        vgg16.dense2.weight = Tensor(checkpointReader.loadTensor(named: "vgg_16/fc7/weights"))
        vgg16.dense2.bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/fc7/weights"))
        vgg16.output.weight = Tensor(checkpointReader.loadTensor(named: "vgg_16/fc8/weights"))
        vgg16.output.bias = Tensor(checkpointReader.loadTensor(named: "vgg_16/fc8/weights"))
        
        return vgg16
    }
}