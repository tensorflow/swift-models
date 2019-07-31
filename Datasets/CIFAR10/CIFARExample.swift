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

public struct CIFARExample: TensorGroup {
    public var label: Tensor<Int32>
    public var data: Tensor<Float>

    public init(label: Tensor<Int32>, data: Tensor<Float>) {
        self.label = label
        self.data = data
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        precondition(_handles.count == 2)
        let labelIndex = _handles.startIndex
        let dataIndex = _handles.index(labelIndex, offsetBy: 1)
        label = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[labelIndex]))
        data = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[dataIndex]))
    }
}
