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

import XCTest
import ModelSupport
import TensorFlow

struct ElementaryFunctionsTests<WrapperTestType: XCTestCase, Reference: ElementaryFunctions, Test: ElementaryFunctions> {
    let createReference: ([Float]) -> Reference
    let referenceToTest: (Reference) -> Test
    let compareReferenceAndTest: (Reference, Test) -> ()
    var rng: SystemRandomNumberGenerator

    public init(
        createReference: @escaping ([Float]) -> Reference,
        referenceToTest: @escaping (Reference) -> Test,
        compareReferenceAndTest: @escaping (Reference, Test) -> (),
        rng: SystemRandomNumberGenerator
    ) {
        self.createReference = createReference
        self.referenceToTest = referenceToTest
        self.compareReferenceAndTest = compareReferenceAndTest
        self.rng = rng
    }

    let functions: [(String, ClosedRange<Float>, (Reference) -> Reference, (Test) -> Test)] = [
        ("sqrt", 0.0...100, Reference.sqrt, Test.sqrt),
        ("cos", -100.0...100, Reference.cos, Test.cos),
        ("sin", -100.0...100, Reference.sin, Test.sin),
        ("tan", -100.0...100, Reference.tan, Test.tan),
        ("acos", -1.0...1.0, Reference.acos, Test.acos),
        ("asin", -1.0...1.0, Reference.asin, Test.asin),
        ("atan", -100.0...100.0, Reference.atan, Test.atan),
        ("cosh", -100.0...100.0, Reference.cosh, Test.cosh),
        ("sinh", -100.0...100.0, Reference.sinh, Test.sinh),
        ("tanh", -100.0...100.0, Reference.tanh, Test.tanh),
        ("acosh", 1.0...100.0, Reference.acosh, Test.acosh),
        ("asinh", -100.0...100.0, Reference.asinh, Test.asinh),
        ("atanh", -1.0...1.0, Reference.atanh, Test.atanh),
        ("exp", -100.0...100.0, Reference.exp, Test.exp),
        ("exp2", -100.0...100.0, Reference.exp2, Test.exp2),
        ("exp10", -100.0...100.0, Reference.exp10, Test.exp10),
        ("expm1", -100.0...100.0, Reference.expm1, Test.expm1),
        ("log", 0.0...100.0, Reference.log, Test.log),
        ("log2", 0.0...100.0, Reference.log2, Test.log2),
        ("log10", 0.0...100.0, Reference.log10, Test.log10),
        ("log1p", -1.0...100.0, Reference.log1p, Test.log1p),
        ("pow", 0.1...100.0, { Reference.pow($0, $0) }, { Test.pow($0, $0) }),
        ("pow", -100.0...100.0, { Reference.pow($0, 2) }, { Test.pow($0, 2) }),
        ("root", -100.0...100.0, { Reference.root($0, 3) }, { Test.root($0, 3) }),
    ]

    mutating func testFunction(range: ClosedRange<Float>, referenceOperator: (Reference) -> Reference, testOperator: (Test) -> Test) {
        let randomNumbers = (0..<10).map { _ in Float.random(in: range, using: &rng) }
        let reference = createReference(randomNumbers)
        let test = referenceToTest(reference)

        compareReferenceAndTest(referenceOperator(reference), testOperator(test))
    }
    
    mutating func testAll() {
        for function in functions {
            testFunction(range: function.1, referenceOperator: function.2, testOperator: function.3)
        }
    }
}

final class AnyLayerTests: XCTestCase {
    func testGradients() {
        var original = Dense<Float>(inputSize: 1, outputSize: 1)
        var erased = AnyLayer(original)

        let originalGradient = gradient(at: original, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let erasedGradient = gradient(at: erased, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        XCTAssertEqual(originalGradient, erasedGradient.base as! Dense<Float>.TangentVector)

        original.move(along: originalGradient)
        erased.move(along: erasedGradient)

        let originalGradient2 = gradient(at: original, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let erasedGradient2 = gradient(at: erased, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        XCTAssertEqual(originalGradient2, erasedGradient2.base as! Dense<Float>.TangentVector)
    }

    func testTangentOperations() {
        let original = Dense<Float>(inputSize: 1, outputSize: 1)
        let erased = AnyLayer(original)

        let originalGradient = gradient(at: original, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let anyLayerGradient = gradient(at: erased, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let transformedOriginal = Dense<Float>.TangentVector.one + originalGradient

        let transformedAny = AnyLayer<Tensor<Float>, Tensor<Float>, Float>.TangentVector.one + anyLayerGradient

        XCTAssertEqual(transformedOriginal, transformedAny.base as! Dense<Float>.TangentVector)
    }

    func testOpaqueScalarEquality() {
        let original = Dense<Float>(inputSize: 1, outputSize: 1)
        let erased = AnyLayer(original)

        XCTAssertEqual(erased.zeroTangentVector, AnyLayerTangentVector(original.zeroTangentVector))
        XCTAssertEqual(AnyLayerTangentVector(original.zeroTangentVector), erased.zeroTangentVector)

        XCTAssertEqual(AnyLayerTangentVector<Float>.one, AnyLayerTangentVector(Dense<Float>.TangentVector.one))
        XCTAssertEqual(AnyLayerTangentVector(Dense<Float>.TangentVector.one), AnyLayerTangentVector<Float>.one)

        XCTAssertEqual(AnyLayerTangentVector<Float>.one.scaled(by: 2.0), AnyLayerTangentVector(Dense<Float>.TangentVector.one.scaled(by: 2.0)))
        XCTAssertEqual(AnyLayerTangentVector(Dense<Float>.TangentVector.one.scaled(by: 2.0)), AnyLayerTangentVector<Float>.one.scaled(by: 2.0))

        XCTAssertNotEqual(AnyLayerTangentVector<Float>.one, AnyLayerTangentVector(Dense<Float>.TangentVector.zero))
        XCTAssertNotEqual(AnyLayerTangentVector(Dense<Float>.TangentVector.zero), AnyLayerTangentVector<Float>.one)
    }

    func testScalarTangentVectorBase() {
        XCTAssertEqual(AnyLayer<Tensor<Float>, Tensor<Float>, Float>.TangentVector.zero.base as! Float, 0)
        XCTAssertEqual(AnyLayer<Tensor<Float>, Tensor<Float>, Float>.TangentVector.one.base as! Float, 1)
        XCTAssertEqual((AnyLayer<Tensor<Float>, Tensor<Float>, Float>.TangentVector.one.scaled(by: 2)).base as! Float, 2)
    }

    func testTangentVectorElementaryFunctions() {
        var generatedTests = ElementaryFunctionsTests<AnyLayerTests, Tensor<Float>, AnyLayerTangentVector<Float>>(
            createReference: { Tensor<Float>($0) },
            referenceToTest: { AnyLayerTangentVector($0) },
            compareReferenceAndTest: { XCTAssertEqual($0, $1.unboxed(as: Tensor<Float>.self)!) },
            rng: SystemRandomNumberGenerator()
        )
        
        generatedTests.testAll()
    }
    
    static var allTests = [
        ("testGradients", testGradients),
        ("testTangentOperations", testTangentOperations),
        ("testOpaqueScalarEquality", testOpaqueScalarEquality),
        ("testScalarTangentVectorBase", testScalarTangentVectorBase),
        ("testTangentVectorElementaryFunctions", testTangentVectorElementaryFunctions),
    ]
}
