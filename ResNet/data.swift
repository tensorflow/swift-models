import Python
import TensorFlow

func maybeDownload(to directory: String = ".") {
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

func loadBatches(from file: String, in directory: String = ".")
    -> (Tensor<Int32>, Tensor<Float>) {
    maybeDownload(to: directory)
    let np = Python.import("numpy")
    let pickle = Python.import("pickle")
    let path = "\(directory)/cifar-10-batches-py/\(file)"
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

    return (Tensor<Int32>(labelTensor), imageTensor / Float(255.0))
}

func loadTrainingBatches() -> (Tensor<Int32>, Tensor<Float>) {
    let data = (1..<6).map { loadBatches(from: "data_batch_\($0)") }
    return (Raw.concat(concatDim: Tensor<Int32>(0), data.map { $0.0 }),
    Raw.concat(concatDim: Tensor<Int32>(0), data.map { $0.1 }))
}

func loadTestBatches() -> (Tensor<Int32>, Tensor<Float>) {
    return loadBatches(from: "test_batch")
}

extension Dataset where Element == TensorPair<Tensor<Int32>, Tensor<Float>> {
    init(fromTuple: (Tensor<Int32>, Tensor<Float>)) {
        self = zip(
            Dataset<Tensor<Int32>>(elements: fromTuple.0),
            Dataset<Tensor<Float>>(elements: fromTuple.1))
    }
}

public func loadCIFAR10() -> (
    Dataset<TensorPair<Tensor<Int32>, Tensor<Float>>>,
    Dataset<TensorPair<Tensor<Int32>, Tensor<Float>>>) {
    let trainingDataset = Dataset<TensorPair<Tensor<Int32>, Tensor<Float>>>(
        fromTuple: loadTrainingBatches())
    let testDataset = Dataset<TensorPair<Tensor<Int32>, Tensor<Float>>>(
        fromTuple: loadTestBatches())
    return (trainingDataset, testDataset)
}
