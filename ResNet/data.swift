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
        let command = "wget -nv -O- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xzf - -C \(directory)"
        subprocess.call(command, shell: true)
    }
}

public struct Example: TensorGroup {
    var label: Tensor<Int32>
    var data: Tensor<Float>
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

    return Example(
        label: Tensor<Int32>(labelTensor), data: imageTensor / Float(255.0))
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

public func loadCIFAR10() -> (Dataset<Example>, Dataset<Example>) {
    let trainingDataset = Dataset<Example>(elements: loadCIFARTrainingFiles())
    let testDataset = Dataset<Example>(elements: loadCIFARTestFile())
    return (trainingDataset, testDataset)
}
