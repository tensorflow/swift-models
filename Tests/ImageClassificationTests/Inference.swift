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
import XCTest

@testable import ImageClassificationModels

final class ImageClassificationInferenceTests: XCTestCase {
    override class func setUp() {
        Context.local.learningPhase = .inference
    }

    func testLeNet() {
        let leNet = LeNet()
        let input = Tensor<Float>(
            randomNormal: [1, 28, 28, 1], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let result = leNet(input)
        XCTAssertEqual(result.shape, [1, 10])
    }

    func testResNet() {
        let inputCIFAR = Tensor<Float>(
            randomNormal: [1, 32, 32, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let resNet18CIFAR = ResNetBasic(inputKind: .resNet18, dataKind: .cifar)
        let resNet18CIFARResult = resNet18CIFAR(inputCIFAR)
        XCTAssertEqual(resNet18CIFARResult.shape, [1, 10])

        let resNet34CIFAR = ResNetBasic(inputKind: .resNet34, dataKind: .cifar)
        let resNet34CIFARResult = resNet34CIFAR(inputCIFAR)
        XCTAssertEqual(resNet34CIFARResult.shape, [1, 10])

        let resNet50CIFAR = ResNet(inputKind: .resNet50, dataKind: .cifar)
        let resNet50CIFARResult = resNet50CIFAR(inputCIFAR)
        XCTAssertEqual(resNet50CIFARResult.shape, [1, 10])

        let resNet101CIFAR = ResNet(inputKind: .resNet101, dataKind: .cifar)
        let resNet101CIFARResult = resNet101CIFAR(inputCIFAR)
        XCTAssertEqual(resNet101CIFARResult.shape, [1, 10])

        let resNet152CIFAR = ResNet(inputKind: .resNet152, dataKind: .cifar)
        let resNet152CIFARResult = resNet152CIFAR(inputCIFAR)
        XCTAssertEqual(resNet152CIFARResult.shape, [1, 10])

        let inputImageNet = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let resNet18ImageNet = ResNetBasic(inputKind: .resNet18, dataKind: .imagenet)
        let resNet18ImageNetResult = resNet18ImageNet(inputImageNet)
        XCTAssertEqual(resNet18ImageNetResult.shape, [1, 1000])

        let resNet34ImageNet = ResNetBasic(inputKind: .resNet34, dataKind: .imagenet)
        let resNet34ImageNetResult = resNet34ImageNet(inputImageNet)
        XCTAssertEqual(resNet34ImageNetResult.shape, [1, 1000])

        let resNet50ImageNet = ResNet(inputKind: .resNet50, dataKind: .imagenet)
        let resNet50ImageNetResult = resNet50ImageNet(inputImageNet)
        XCTAssertEqual(resNet50ImageNetResult.shape, [1, 1000])

        let resNet101ImageNet = ResNet(inputKind: .resNet101, dataKind: .imagenet)
        let resNet101ImageNetResult = resNet101ImageNet(inputImageNet)
        XCTAssertEqual(resNet101ImageNetResult.shape, [1, 1000])

        let resNet152ImageNet = ResNet(inputKind: .resNet152, dataKind: .imagenet)
        let resNet152ImageNetResult = resNet152ImageNet(inputImageNet)
        XCTAssertEqual(resNet152ImageNetResult.shape, [1, 1000])
    }

    func testResNetV2() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let resNet18ImageNet = PreActivatedResNet18(imageSize: 224, classCount: 1000)
        let resNet18ImageNetResult = resNet18ImageNet(input)
        XCTAssertEqual(resNet18ImageNetResult.shape, [1, 1000])

        let resNet34ImageNet = PreActivatedResNet34(imageSize: 224, classCount: 1000)
        let resNet34ImageNetResult = resNet34ImageNet(input)
        XCTAssertEqual(resNet34ImageNetResult.shape, [1, 1000])
    }

    func testSqueezeNet() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let squeezeNet = SqueezeNet(classCount: 1000)
        let squeezeNetResult = squeezeNet(input)
        XCTAssertEqual(squeezeNetResult.shape, [1, 1000])
    }

    func testWideResNet() {
        let input = Tensor<Float>(
            randomNormal: [1, 32, 32, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let wideResNet16 = WideResNet(kind: .wideResNet16)
        let wideResNet16Result = wideResNet16(input)
        XCTAssertEqual(wideResNet16Result.shape, [1, 10])

        let wideResNet16k10 = WideResNet(kind: .wideResNet16k10)
        let wideResNet16k10Result = wideResNet16k10(input)
        XCTAssertEqual(wideResNet16k10Result.shape, [1, 10])

        let wideResNet22 = WideResNet(kind: .wideResNet22)
        let wideResNet22Result = wideResNet22(input)
        XCTAssertEqual(wideResNet22Result.shape, [1, 10])

        let wideResNet22k10 = WideResNet(kind: .wideResNet22k10)
        let wideResNet22k10Result = wideResNet22k10(input)
        XCTAssertEqual(wideResNet22k10Result.shape, [1, 10])

        let wideResNet28 = WideResNet(kind: .wideResNet28)
        let wideResNet28Result = wideResNet28(input)
        XCTAssertEqual(wideResNet28Result.shape, [1, 10])

        let wideResNet28k12 = WideResNet(kind: .wideResNet28k12)
        let wideResNet28k12Result = wideResNet28k12(input)
        XCTAssertEqual(wideResNet28k12Result.shape, [1, 10])

        let wideResNet40k1 = WideResNet(kind: .wideResNet40k1)
        let wideResNet40k1Result = wideResNet40k1(input)
        XCTAssertEqual(wideResNet40k1Result.shape, [1, 10])

        let wideResNet40k2 = WideResNet(kind: .wideResNet40k2)
        let wideResNet40k2Result = wideResNet40k2(input)
        XCTAssertEqual(wideResNet40k2Result.shape, [1, 10])

        let wideResNet40k4 = WideResNet(kind: .wideResNet40k4)
        let wideResNet40k4Result = wideResNet40k4(input)
        XCTAssertEqual(wideResNet40k4Result.shape, [1, 10])

        let wideResNet40k8 = WideResNet(kind: .wideResNet40k8)
        let wideResNet40k8Result = wideResNet40k8(input)
        XCTAssertEqual(wideResNet40k8Result.shape, [1, 10])
    }
}

extension ImageClassificationInferenceTests {
    static var allTests = [
        ("testLeNet", testLeNet),
        ("testResNet", testResNet),
        ("testResNetV2", testResNetV2),
        ("testSqueezeNet", testSqueezeNet),
        ("testWideResNet", testWideResNet),
    ]
}
