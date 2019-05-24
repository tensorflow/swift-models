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

// TODO: Remove this when it's moved to the standard library.
extension Array where Element: Differentiable {
    @differentiable(wrt: (self, initialResult), vjp: reduceDerivative)
    func differentiableReduce<Result: Differentiable>(
        _ initialResult: Result,
        _ nextPartialResult: @differentiable (Result, Element) -> Result
    ) -> Result {
        return reduce(initialResult, nextPartialResult)
    }
    
    func reduceDerivative<Result: Differentiable>(
        _ initialResult: Result,
        _ nextPartialResult: @differentiable (Result, Element) -> Result
    ) -> (Result, (Result.TangentVector) -> (Array.TangentVector, Result.TangentVector)) {
        var pullbacks: [(Result.TangentVector)
            -> (Result.TangentVector, Element.TangentVector)] = []
        let count = self.count
        pullbacks.reserveCapacity(count)
        var result = initialResult
        for element in self {
            let (y, pb) = Swift.valueWithPullback(at: result, element, in: nextPartialResult)
            result = y
            pullbacks.append(pb)
        }
        return (value: result, pullback: { cotangent in
            var resultCotangent = cotangent
            var elementCotangents = TangentVector([])
            elementCotangents.base.reserveCapacity(count)
            for pullback in pullbacks.reversed() {
                let (newResultCotangent, elementCotangent) = pullback(resultCotangent)
                resultCotangent = newResultCotangent
                elementCotangents.base.append(elementCotangent)
            }
            return (TangentVector(elementCotangents.base.reversed()), resultCotangent)
        })
    }
}
