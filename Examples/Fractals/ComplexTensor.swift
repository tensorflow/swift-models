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

import ArgumentParser
import TensorFlow

struct ComplexTensor {
  let real: Tensor<Float>
  let imaginary: Tensor<Float>
}

func +(lhs: ComplexTensor, rhs: ComplexTensor) -> ComplexTensor {
  let real = lhs.real + rhs.real
  let imaginary = lhs.imaginary + rhs.imaginary
  return ComplexTensor(real: real, imaginary: imaginary)
}

func *(lhs: ComplexTensor, rhs: ComplexTensor) -> ComplexTensor {
  let real = lhs.real .* rhs.real - lhs.imaginary .* rhs.imaginary
  let imaginary = lhs.real .* rhs.imaginary + lhs.imaginary .* rhs.real
  return ComplexTensor(real: real, imaginary: imaginary)
}

func abs(_ value: ComplexTensor) -> Tensor<Float> {
  return value.real .* value.real + value.imaginary .* value.imaginary
}

struct ComplexRegion {
    let realMinimum: Float
    let realMaximum: Float
    let imaginaryMinimum: Float
    let imaginaryMaximum: Float
}

extension ComplexRegion: ExpressibleByArgument {
    init?(argument: String) {
        let subArguments = argument.split(separator: ",").compactMap { Float(String($0)) }
        guard subArguments.count >= 4 else { return nil }
        
        self.realMinimum = subArguments[0]
        self.realMaximum = subArguments[1]
        self.imaginaryMinimum = subArguments[2]
        self.imaginaryMaximum = subArguments[3]
    }

    var defaultValueDescription: String {
        "\(self.realMinimum) \(self.realMaximum) \(self.imaginaryMinimum) \(self.imaginaryMaximum)"
    }
}
