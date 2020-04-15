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

    func testDenseNet121() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let denseNet121 = DenseNet121(classCount: 1000)
        let denseNet121Result = denseNet121(input)
        XCTAssertEqual(denseNet121Result.shape, [1, 1000])
    }

    func testEfficientNet() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let efficientNet = EfficientNet(classCount: 1000)
        let efficientNetResult = efficientNet(input)
        XCTAssertEqual(efficientNetResult.shape, [1, 1000])

        let efficientNetSmall = EfficientNet(classCount: 10)
        let efficientNetSmallResult = efficientNetSmall(input)
        XCTAssertEqual(efficientNetSmallResult.shape, [1, 10])

        let efficientNetB0 = EfficientNet(kind: .efficientnetB0, classCount: 1000)
        let efficientNetB0Result = efficientNetB0(input)
        XCTAssertEqual(efficientNetB0Result.shape, [1, 1000])

        let efficientNetB1 = EfficientNet(kind: .efficientnetB1, classCount: 1000)
        let efficientNetB1Result = efficientNetB1(input)
        XCTAssertEqual(efficientNetB1Result.shape, [1, 1000])

        let efficientNetB2 = EfficientNet(kind: .efficientnetB2, classCount: 1000)
        let efficientNetB2Result = efficientNetB2(input)
        XCTAssertEqual(efficientNetB2Result.shape, [1, 1000])

        let efficientNetB3 = EfficientNet(kind: .efficientnetB3, classCount: 1000)
        let efficientNetB3Result = efficientNetB3(input)
        XCTAssertEqual(efficientNetB3Result.shape, [1, 1000])

        let efficientNetB4 = EfficientNet(kind: .efficientnetB4, classCount: 1000)
        let efficientNetB4Result = efficientNetB4(input)
        XCTAssertEqual(efficientNetB4Result.shape, [1, 1000])

        let efficientNetB5 = EfficientNet(kind: .efficientnetB5, classCount: 1000)
        let efficientNetB5Result = efficientNetB5(input)
        XCTAssertEqual(efficientNetB5Result.shape, [1, 1000])

        let efficientNetB6 = EfficientNet(kind: .efficientnetB6, classCount: 1000)
        let efficientNetB6Result = efficientNetB6(input)
        XCTAssertEqual(efficientNetB6Result.shape, [1, 1000])

        let efficientNetB7 = EfficientNet(kind: .efficientnetB7, classCount: 1000)
        let efficientNetB7Result = efficientNetB7(input)
        XCTAssertEqual(efficientNetB7Result.shape, [1, 1000])

        let efficientNetB8 = EfficientNet(kind: .efficientnetB8, classCount: 1000)
        let efficientNetB8Result = efficientNetB8(input)
        XCTAssertEqual(efficientNetB8Result.shape, [1, 1000])

        let efficientNetL2 = EfficientNet(kind: .efficientnetL2, classCount: 1000)
        let efficientNetL2Result = efficientNetL2(input)
        XCTAssertEqual(efficientNetL2Result.shape, [1, 1000])
    }

    func testLeNet() {
        let leNet = LeNet()
        let input = Tensor<Float>(
            randomNormal: [1, 28, 28, 1], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let result = leNet(input)
        XCTAssertEqual(result.shape, [1, 10])
    }

    func testMobileNetV1() {
        // ImageNet size
        let inputImageNet = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let mobileNet = MobileNetV1(classCount: 1000)
        let mobileNetResult = mobileNet(inputImageNet)
        XCTAssertEqual(mobileNetResult.shape, [1, 1000])

        // CIFAR10 size
        let inputCIFAR = Tensor<Float>(
            randomNormal: [1, 32, 32, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let mobileNetCIFAR = MobileNetV1(classCount: 10)
        let mobileNetCIFARResult = mobileNetCIFAR(inputCIFAR)
        XCTAssertEqual(mobileNetCIFARResult.shape, [1, 10])

        // Width multiplier
        let mobileNetWMSmall = MobileNetV1(classCount: 10, widthMultiplier: 0.5)
        let mobileNetWMSmallResult = mobileNetWMSmall(inputCIFAR)
        XCTAssertEqual(mobileNetWMSmallResult.shape, [1, 10])

        // Width multiplier
        let mobileNetWMLarge = MobileNetV1(classCount: 10, widthMultiplier: 1.5)
        let mobileNetWMLargeResult = mobileNetWMLarge(inputCIFAR)
        XCTAssertEqual(mobileNetWMLargeResult.shape, [1, 10])

        // Depth multiplier and dropout
        let mobileNetDMD = MobileNetV1(classCount: 10, depthMultiplier: 2, dropout: 0.01)
        let mobileNetDMDResult = mobileNetDMD(inputCIFAR)
        XCTAssertEqual(mobileNetDMDResult.shape, [1, 10])
    }

    func testMobileNetV2() {
        // ImageNet size
        let inputImageNet = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let mobileNet = MobileNetV2(classCount: 1000)
        let mobileNetResult = mobileNet(inputImageNet)
        XCTAssertEqual(mobileNetResult.shape, [1, 1000])

        // Width multiplier
        let mobileNet2WMSmall = MobileNetV2(classCount: 10, widthMultiplier: 0.5)
        let mobileNet2WMSmallResult = mobileNet2WMSmall(inputImageNet)
        XCTAssertEqual(mobileNet2WMSmallResult.shape, [1, 10])

        // Width multiplier
        let mobileNet2WMLarge = MobileNetV2(classCount: 10, widthMultiplier: 1.4)
        let mobileNet2WMLargeResult = mobileNet2WMLarge(inputImageNet)
        XCTAssertEqual(mobileNet2WMLargeResult.shape, [1, 10])
    }

    func testMobileNetV3() {
        // ImageNet size
        let inputImageNet = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let mobileNetLarge = MobileNetV3Large(classCount: 10)
        let mobileNetLargeResult = mobileNetLarge(inputImageNet)
        XCTAssertEqual(mobileNetLargeResult.shape, [1, 10])

        let mobileNetSmall = MobileNetV3Small(classCount: 10)
        let mobileNetSmallResult = mobileNetSmall(inputImageNet)
        XCTAssertEqual(mobileNetSmallResult.shape, [1, 10])

        let mobileNetLargeAndWide = MobileNetV3Large(classCount: 10, widthMultiplier: 1.4)
        let mobileNetLargeAndWideResult = mobileNetLargeAndWide(inputImageNet)
        XCTAssertEqual(mobileNetLargeAndWideResult.shape, [1, 10])

        let mobileNetLargeAndThin = MobileNetV3Large(classCount: 10, widthMultiplier: 0.5)
        let mobileNetLargeAndThinResult = mobileNetLargeAndThin(inputImageNet)
        XCTAssertEqual(mobileNetLargeAndThinResult.shape, [1, 10])

        let mobileNetSmallAndWide = MobileNetV3Small(classCount: 10, widthMultiplier: 1.4)
        let mobileNetSmallAndWideResult = mobileNetSmallAndWide(inputImageNet)
        XCTAssertEqual(mobileNetSmallAndWideResult.shape, [1, 10])

        let mobileNetSmallAndThin = MobileNetV3Small(classCount: 10, widthMultiplier: 0.5)
        let mobileNetSmallAndThinResult = mobileNetSmallAndThin(inputImageNet)
        XCTAssertEqual(mobileNetSmallAndThinResult.shape, [1, 10])
    }

    func testResNet() {
        let inputCIFAR = Tensor<Float>(
            randomNormal: [1, 32, 32, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let resNet18CIFAR = ResNet(classCount: 10, depth: .resNet18, downsamplingInFirstStage: true)
        let resNet18CIFARResult = resNet18CIFAR(inputCIFAR)
        XCTAssertEqual(resNet18CIFARResult.shape, [1, 10])

        let resNet34CIFAR = ResNet(classCount: 10, depth: .resNet34, downsamplingInFirstStage: true)
        let resNet34CIFARResult = resNet34CIFAR(inputCIFAR)
        XCTAssertEqual(resNet34CIFARResult.shape, [1, 10])

        let resNet50CIFARV1 = ResNet(
            classCount: 10, depth: .resNet50, downsamplingInFirstStage: true, useLaterStride: false)
        let resNet50CIFARV1Result = resNet50CIFARV1(inputCIFAR)
        XCTAssertEqual(resNet50CIFARV1Result.shape, [1, 10])

        let resNet50CIFAR = ResNet(classCount: 10, depth: .resNet50, downsamplingInFirstStage: true)
        let resNet50CIFARResult = resNet50CIFAR(inputCIFAR)
        XCTAssertEqual(resNet50CIFARResult.shape, [1, 10])

        let resNet101CIFAR = ResNet(
            classCount: 10, depth: .resNet101, downsamplingInFirstStage: true)
        let resNet101CIFARResult = resNet101CIFAR(inputCIFAR)
        XCTAssertEqual(resNet101CIFARResult.shape, [1, 10])

        let resNet152CIFAR = ResNet(
            classCount: 10, depth: .resNet152, downsamplingInFirstStage: true)
        let resNet152CIFARResult = resNet152CIFAR(inputCIFAR)
        XCTAssertEqual(resNet152CIFARResult.shape, [1, 10])

        let inputImageNet = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let resNet18ImageNet = ResNet(classCount: 1000, depth: .resNet18)
        let resNet18ImageNetResult = resNet18ImageNet(inputImageNet)
        XCTAssertEqual(resNet18ImageNetResult.shape, [1, 1000])

        let resNet34ImageNet = ResNet(classCount: 1000, depth: .resNet34)
        let resNet34ImageNetResult = resNet34ImageNet(inputImageNet)
        XCTAssertEqual(resNet34ImageNetResult.shape, [1, 1000])

        let resNet50ImageNetV1 = ResNet(classCount: 1000, depth: .resNet50, useLaterStride: false)
        let resNet50ImageNetV1Result = resNet50ImageNetV1(inputImageNet)
        XCTAssertEqual(resNet50ImageNetV1Result.shape, [1, 1000])

        let resNet50ImageNet = ResNet(classCount: 1000, depth: .resNet50)
        let resNet50ImageNetResult = resNet50ImageNet(inputImageNet)
        XCTAssertEqual(resNet50ImageNetResult.shape, [1, 1000])

        let resNet101ImageNet = ResNet(classCount: 1000, depth: .resNet101)
        let resNet101ImageNetResult = resNet101ImageNet(inputImageNet)
        XCTAssertEqual(resNet101ImageNetResult.shape, [1, 1000])

        let resNet152ImageNet = ResNet(classCount: 1000, depth: .resNet152)
        let resNet152ImageNetResult = resNet152ImageNet(inputImageNet)
        XCTAssertEqual(resNet152ImageNetResult.shape, [1, 1000])
    }

    func testResNetV2() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))

        let resNet18 = ResNetV2(classCount: 1000, depth: .resNet18)
        let resNet18Result = resNet18(input)
        XCTAssertEqual(resNet18Result.shape, [1, 1000])

        let resNet34 = ResNetV2(classCount: 1000, depth: .resNet34)
        let resNet34Result = resNet34(input)
        XCTAssertEqual(resNet34Result.shape, [1, 1000])

        let resNet50 = ResNetV2(classCount: 1000, depth: .resNet50)
        let resNet50Result = resNet50(input)
        XCTAssertEqual(resNet50Result.shape, [1, 1000])

        let resNet101 = ResNetV2(classCount: 1000, depth: .resNet101)
        let resNet101Result = resNet101(input)
        XCTAssertEqual(resNet101Result.shape, [1, 1000])

        let resNet152 = ResNetV2(classCount: 1000, depth: .resNet152)
        let resNet152Result = resNet152(input)
        XCTAssertEqual(resNet152Result.shape, [1, 1000])
    }

    func testSqueezeNetV1_0() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let squeezeNet = SqueezeNetV1_0(classCount: 1000)
        let squeezeNetResult = squeezeNet(input)
        XCTAssertEqual(squeezeNetResult.shape, [1, 1000])
    }

    func testSqueezeNetV1_1() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let squeezeNet = SqueezeNetV1_1(classCount: 1000)
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

    func testVGG16() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let vgg16 = VGG16(classCount: 1000)
        let vgg16Result = vgg16(input)
        XCTAssertEqual(vgg16Result.shape, [1, 1000])
    }

    func testVGG19() {
        let input = Tensor<Float>(
            randomNormal: [1, 224, 224, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let vgg19 = VGG19(classCount: 1000)
        let vgg19Result = vgg19(input)
        XCTAssertEqual(vgg19Result.shape, [1, 1000])
    }
}

extension ImageClassificationInferenceTests {
    static var allTests = [
        ("testDenseNet121", testDenseNet121),
        ("testEfficientNet", testEfficientNet),
        ("testLeNet", testLeNet),
        ("testMobileNetV1", testMobileNetV1),
        ("testMobileNetV2", testMobileNetV2),
        ("testMobileNetV3", testMobileNetV3),
        ("testResNet", testResNet),
        ("testResNetV2", testResNetV2),
        ("testSqueezeNetV1_0", testSqueezeNetV1_0),
        ("testSqueezeNetV1_1", testSqueezeNetV1_1),
        ("testWideResNet", testWideResNet),
        ("testVGG16", testVGG16),
        ("testVGG19", testVGG19),
    ]
}
