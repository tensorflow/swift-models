import Dispatch
import TensorFlow
import Datasets
import PythonKit
import ImageClassificationModels

let tqdm = Python.import("tqdm")

let np = Python.import("numpy")

class GpuWorkerInput<Model: Layer> {
    var inSemaphore: DispatchSemaphore
    var outSemaphore: DispatchSemaphore
    var device: UInt
    var model: Model?
    var finished = false
    var x: Model.Input?
    var y: Model.Output?
    var output: Model.TangentVector?
    var loss: Float?

    init(inSemaphore: DispatchSemaphore,
         outSemaphore: DispatchSemaphore,
         device: UInt,
         model: Model? = nil,
         finished: Bool = false,
         x: Model.Input? = nil,
         y: Model.Output? = nil,
         output: Model.TangentVector? = nil,
         loss: Float? = nil) {
        self.inSemaphore = inSemaphore
        self.outSemaphore = outSemaphore
        self.device = device
        self.model = model
        self.finished = finished
        self.x = x
        self.y = y
        self.output = output
        self.loss = loss
    }
}

func gpuWorker<Model: Layer>(_ workerInput: GpuWorkerInput<Model>, loss: @differentiable (Model, Model.Input, Model.Output) -> (Tensor<Float>)) {
    withDevice(.cpu, workerInput.device) {
        let gpu = Device(kind: .CPU, ordinal: 0, backend: .XLA)
        while true {
            workerInput.inSemaphore.wait()
            if workerInput.finished {
                break
            }

            let inputData = workerInput.x!
            let outputData = workerInput.y!
            let (loss, grad) = TensorFlow.valueWithGradient(at: workerInput.model!) { model -> Tensor<Float> in
                return loss(model, inputData, outputData)
            }
            LazyTensorBarrier(on: gpu, wait: true)
            workerInput.output = grad
            workerInput.loss = loss.scalarized()

            workerInput.outSemaphore.signal()
        }
    }
}

func averageTangentVectors(_ tangentVectors: [UNet.TangentVector]) -> UNet.TangentVector {
    withDevice(.cpu) {
        var avg = UNet.TangentVector.zero
        let divisor = Tensor(copying: Tensor<Float>(Float(tangentVectors.count)), to: cpuDevice)
        for kp in avg.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            avg[keyPath: kp] = Tensor(copying: avg[keyPath: kp], to: cpuDevice)
            for v in tangentVectors {
                avg[keyPath: kp] += Tensor(copying: v[keyPath: kp], to: cpuDevice)
            }
            avg[keyPath: kp] /= divisor
        }
        return avg
    }
}

// This function is to be removed when swift-apis issue #1038 is resolved.
@differentiable(wrt: logits)
public func tbSigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(
  logits: Tensor<Scalar>,
  labels: Tensor<Scalar>,
  reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean
) -> Tensor<Scalar> {
  let device = logits.device
  // This numerically stable implementation is based on the TensorFlow Python API.
  let maxLogitsWithZero = max(logits, Tensor(0, on: device))
  let negAbsLogits = max(logits, -logits)  // Custom `abs` to compute gradients at `0`.
  return reduction(maxLogitsWithZero - logits * labels + log1p(exp(-negAbsLogits)))
}

@differentiable
func unetWorkerLoss(model: UNet, input: Tensor<Float>, output: Tensor<Float>) -> Tensor<Float> {
    tbSigmoidCrossEntropy(logits: model(input).reshaped(to: [-1, 256 * 256]), labels: output.reshaped(to: [-1, 256 * 256]))
}

let gpus = 1
let batchSize = 16
let cpuDevice = Device(kind: .CPU, ordinal: 0, backend: .TF_EAGER)

let dataset = OxfordIIITPets<SystemRandomNumberGenerator>(batchSize: batchSize * gpus, imageSize: 256, on: cpuDevice)

var workerData: [GpuWorkerInput<UNet>] = []
for gpu in 0..<gpus {
    let worker: GpuWorkerInput<UNet> = .init(inSemaphore: DispatchSemaphore(value: 0),
                                             outSemaphore: DispatchSemaphore(value: 0),
                                             device: UInt(gpu))
    DispatchQueue.global().async {
        gpuWorker(worker, loss: unetWorkerLoss)
    }
    workerData.append(worker)
}

var model = UNet()
model.move(to: cpuDevice)

var opt = Adam(copying: Adam(for: model, learningRate: 0.0001 * Float(gpus)), to: cpuDevice)
for (epoch, epochBatches) in dataset.training.prefix(10).enumerated() {
    print("Epoch \(epoch)")
    let pbar = tqdm.tqdm(total: epochBatches.count)
    defer { pbar.close() }

    Context.local.learningPhase = .training
    var losses: [Float] = []
    for batch in epochBatches {
        defer { pbar.update(1) }

        for (workerIdx, worker) in workerData.enumerated() {
            let start = workerIdx * batchSize
            let end = start + batchSize
            worker.model = model
            worker.x = batch.data[start..<end]
            worker.y = Tensor<Float>(batch.label[start..<end].clipped(min: 0, max: 1))
            worker.inSemaphore.signal()
        }

        for worker in workerData {
            worker.outSemaphore.wait()
        }

        let grad = averageTangentVectors(workerData.map { $0.output! })
        opt.update(&model, along: grad)
        LazyTensorBarrier(on: cpuDevice)
        losses += workerData.map { $0.loss! }
    }
    print("Loss: \(losses.reduce(0, +) / Float(losses.count))")

    np.save("weights.npy", model.weights())
}
