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

import Python
import TensorFlow

/// Use Python and shell calls to download and extract the CIFAR-10 tarball if not already done
/// This can fail for many reasons (e.g. lack of `wget`, `tar`, or an Internet connection)
func downloadCIFAR10IfNotPresent(to directory: String = ".") {
    let subprocess = Python.import("subprocess")
    let path = Python.import("os.path")
    let filepath = "\(directory)/cifar-10-batches-py"
    let isdir = Bool(path.isdir(filepath))!
    if !isdir {
        print("Downloading CIFAR data...")
        let command = """
            wget -nv -O- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xzf - \
            -C \(directory)
            """
        subprocess.call(command, shell: true)
    }
}

extension Tensor where Scalar : _TensorFlowDataTypeCompatible {
    public var _tfeTensorHandle: _AnyTensorHandle {
        TFETensorHandle(_owning: handle._cTensorHandle)
    }
}

struct Example: TensorGroup {
    var label: Tensor<Int32>
    var data: Tensor<Float>

    init(label: Tensor<Int32>, data: Tensor<Float>) {
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

    public var _tensorHandles: [_AnyTensorHandle] { [label._tfeTensorHandle, data._tfeTensorHandle] }
}

// Each CIFAR data file is provided as a Python pickle of NumPy arrays
func loadCIFARFile(named name: String, in directory: String = ".") -> Example {
    downloadCIFAR10IfNotPresent(to: directory)
    let np = Python.import("numpy")
    let pickle = Python.import("pickle")
    let path = "\(directory)/cifar-10-batches-py/\(name)"
    let f = Python.open(path, "rb")
    let res = pickle.load(f, encoding: "bytes")

    let bytes = res[Python.bytes("data", encoding: "utf8")]
    let labels = res[Python.bytes("labels", encoding: "utf8")]

    let labelTensor = Tensor<Int64>(numpy: np.array(labels))!
    let images = Tensor<UInt8>(numpy: bytes)!
    let imageCount = images.shape[0]

    // reshape and transpose from the provided N(CHW) to TF default NHWC
    let imageTensor = Tensor<Float>(images
        .reshaped(to: [imageCount, 3, 32, 32])
        .transposed(withPermutations: [0, 2, 3, 1]))

    let mean = Tensor<Float>([0.485, 0.456, 0.406])
    let std  = Tensor<Float>([0.229, 0.224, 0.225])
    let imagesNormalized = ((imageTensor / 255.0) - mean) / std

    return Example(label: Tensor<Int32>(labelTensor), data: imagesNormalized)
}

func loadCIFARTrainingFiles() -> Example {
    let data = (1..<6).map { loadCIFARFile(named: "data_batch_\($0)") }
    return Example(
        label: Raw.concat(concatDim: Tensor<Int32>(0), data.map { $0.label }),
        data: Raw.concat(concatDim: Tensor<Int32>(0), data.map { $0.data })
    )
}

func loadCIFARTestFile() -> Example {
    return loadCIFARFile(named: "test_batch")
}

func loadCIFAR10() -> (training: Dataset<Example>, test: Dataset<Example>) {
    let trainingDataset = Dataset<Example>(elements: loadCIFARTrainingFiles())
    let testDataset = Dataset<Example>(elements: loadCIFARTestFile())
    return (training: trainingDataset, test: testDataset)
}
