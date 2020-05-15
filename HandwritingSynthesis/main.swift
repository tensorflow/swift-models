import Foundation
import TensorFlow
import PythonKit

let h5py = Python.import("h5py")
let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")
let matplotlib = Python.import("matplotlib")

func multinomial(_ x: [Float]) -> Int {
    let y = Float.random(in: 0...1)
    var z: Float = 0
    for (iIdx, i) in x.enumerated() {
        z += i
        if z >= y {
            return iIdx
        }
    }
    fatalError()
}

extension Tensor {
    mutating func scatter(indices: Tensor<Int32>, scalar: Tensor<Scalar>) {
        for i in 0..<shape[0] {
            for j in 0..<shape[1] {
                self[i][j][Int(indices[i][j].scalar!)] = scalar
            }
        }
    }
}

struct MixtureGaussians2DAndPen {
    struct Input {
        var h: Tensor<Float>
        var bias: Tensor<Float> = Tensor(0.0)
    }
    
    struct Parameters {
        var pi: Tensor<Float>
        var meanX: Tensor<Float>
        var meanY: Tensor<Float>
        var stdX: Tensor<Float>
        var stdY: Tensor<Float>
        var rho: Tensor<Float>
        var bernoulli: Tensor<Float>
    }
    
    let eps: Tensor<Float> = Tensor(1e-6)
    let mixtureComponents: Int
    
    var linear: Dense<Float>
    
    init(inputs: Int, mixtureComponents: Int) {
        self.mixtureComponents = mixtureComponents
        self.linear = Dense(inputSize: inputs, outputSize: mixtureComponents * 6 + 1)
    }
    
    func computeParameters(h: Tensor<Float>, bias: Tensor<Float>) -> Parameters {
        let yHat = linear(h)
        let M = self.mixtureComponents
        var pi = yHat.slice(lowerBounds: [0, 0], upperBounds: [yHat.shape[0], M])
        pi = pi * (1 + bias)
        pi = pi - pi.max()
        pi = softmax(pi, alongAxis: 1)
        let meanX = yHat.slice(lowerBounds: [0, M], upperBounds: [yHat.shape[0], M * 2])
        let meanY = yHat.slice(lowerBounds: [0, M * 2], upperBounds: [yHat.shape[0], M * 3])
        let stdX = exp(yHat.slice(lowerBounds: [0, M * 3], upperBounds: [yHat.shape[0], M * 4]) - bias) + eps
        let stdY = exp(yHat.slice(lowerBounds: [0, M * 4], upperBounds: [yHat.shape[0], M * 5]) - bias) + eps
        var rho = tanh(yHat.slice(lowerBounds: [0, M * 5], upperBounds: [yHat.shape[0], M * 6]))
        rho = rho / (1 + eps)
        var bernoulli = sigmoid(yHat.slice(lowerBounds: [0, M * 6], upperBounds: [yHat.shape[0], M * 6 + 1]))
        bernoulli = (bernoulli + eps) / (1 + 2 * eps)
        return .init(pi: pi, meanX: meanX, meanY: meanY, stdX: stdX, stdY: stdY, rho: rho, bernoulli: bernoulli)
    }
    
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        let parameters = computeParameters(h: input.h, bias: input.bias)
        
        let mode = multinomial(parameters.pi.flattened().scalars)
        
        let mx = parameters.meanX.transposed()[mode]
        let my = parameters.meanY.transposed()[mode]
        let sx = parameters.stdX.transposed()[mode]
        let sy = parameters.stdY.transposed()[mode]
        let r = parameters.rho.transposed()[mode]
        
        let x = Tensor<Float>(randomNormal: [input.h.shape[0]])
        let y = Tensor<Float>(randomNormal: [input.h.shape[0]])
        
        let xn = (mx + sx * x).reshaped(to: [-1, 1])
        let p1 = pow(Tensor(1) - pow(r, 2), 0.5)
        let p2 = (x * r + y * p1)
        let yn = (my + sy * p2).reshaped(to: [-1, 1])
        
        let uniform = Tensor<Float>(randomUniform: [input.h.shape[0]]).scalars
        let pen = Tensor<Float>(parameters.bernoulli.flattened().scalars.enumerated().map { x -> Float in x.element > uniform[x.offset] ? 1 : 0 }).reshaped(to: [-1, 1])
        
        return xn.concatenated(with: yn, alongAxis: 1).concatenated(with: pen, alongAxis: 1)
    }
}

struct Attention {
    struct Input {
        var h1: Tensor<Float>
        var kappaPrev: Tensor<Float>
        var cSeqLen: Float
    }
    
    struct Output {
        var phi: Tensor<Float>
        var kappa: Tensor<Float>
    }
    
    let mixtureComponents: Int
    var linear: Dense<Float>
    
    init(inputs: Int, mixtureComponents: Int) {
        self.mixtureComponents = mixtureComponents
        self.linear = Dense(inputSize: inputs, outputSize: mixtureComponents * 3)
    }
    
    func callAsFunction(_ input: Input) -> Output {
        let K = self.mixtureComponents
        let params = exp(linear(input.h1))
        let alpha = params.slice(lowerBounds: [0, 0], upperBounds: [params.shape[0], K])
        let beta = params.slice(lowerBounds: [0, K], upperBounds: [params.shape[0], K * 2])
        let kappa = input.kappaPrev + 0.1 * params.slice(lowerBounds: [0, K * 2], upperBounds: [params.shape[0], params.shape[1]])
        let u = Tensor<Float>(rangeFrom: 0, to: input.cSeqLen, stride: 1).reshaped(to: [-1, 1, 1])
        let phi = (alpha * exp(-beta * pow((kappa - u), 2))).sum(squeezingAxes: -1)
        return .init(phi: phi, kappa: kappa)
    }
}

struct PyTorchLSTMCell {
    struct Input {
        var input: Tensor<Float>
        var state: State
    }
    
    struct State {
        var hidden: Tensor<Float>
        var cell: Tensor<Float>
    }
    
    let hiddenSize: Int
    
    var ih: Dense<Float>
    var hh: Dense<Float>
    
    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        self.ih = Dense(inputSize: inputSize, outputSize: hiddenSize * 4)
        self.hh = Dense(inputSize: hiddenSize, outputSize: hiddenSize * 4)
    }
    
    func callAsFunction(_ input: Input) -> State {
        let batchSize = input.input.shape[0]
        let (hx, cx) = (input.state.hidden, input.state.cell)
        let gates = ih(input.input) + hh(hx)
        
        var inGate = gates.slice(lowerBounds: [0, 0], upperBounds: [batchSize, hiddenSize])
        var forgetGate = gates.slice(lowerBounds: [0, hiddenSize], upperBounds: [batchSize, hiddenSize * 2])
        var cellGate = gates.slice(lowerBounds: [0, hiddenSize * 2], upperBounds: [batchSize, hiddenSize * 3])
        var outGate = gates.slice(lowerBounds: [0, hiddenSize * 3], upperBounds: [batchSize, hiddenSize * 4])
        
        inGate = sigmoid(inGate)
        forgetGate = sigmoid(forgetGate)
        cellGate = tanh(cellGate)
        outGate = sigmoid(outGate)
        
        let cy = (forgetGate * cx) + (inGate * cellGate)
        let hy = outGate * tanh(cy)
        
        return .init(hidden: hy, cell: cy)
    }
}

struct HandwritingGenerator {
    var attention: Attention
    var rnnCell: PyTorchLSTMCell
    var mixture: MixtureGaussians2DAndPen
    
    init() {
        attention = Attention(inputs: 900, mixtureComponents: 10)
        rnnCell = PyTorchLSTMCell(inputSize: 81, hiddenSize: 900)
        mixture = MixtureGaussians2DAndPen(inputs: 900 + 81, mixtureComponents: 20)
    }
    
    func rnnStep(inputs: Tensor<Float>, hStatePre: PyTorchLSTMCell.State, kPre: Tensor<Float>, wPre: Tensor<Float>, c: Tensor<Float>, cMask: Tensor<Float>, mask: Tensor<Float>) -> (PyTorchLSTMCell.State, Tensor<Float>, Tensor<Float>, Tensor<Float>) {
        let inputs = inputs.concatenated(with: wPre, alongAxis: 1)
        
        let hState = rnnCell.callAsFunction(.init(input: inputs, state: hStatePre))
        let h = hState.hidden
        
        let attOutput = attention(.init(h1: h, kappaPrev: kPre, cSeqLen: Float(c.shape[0])))
        var (phi, k) = (attOutput.phi, attOutput.kappa)
        phi = phi * cMask
        
        var w = (phi.reshaped(to: [phi.shape[0], phi.shape[1], -1]) * c).sum(squeezingAxes: 0)
        k = mask.reshaped(to: [-1, 1]) * k + (1 - mask.reshaped(to: [-1, 1])) * kPre
        w = mask.reshaped(to: [-1, 1]) * w + (1 - mask.reshaped(to: [-1, 1])) * wPre
        
        return (hState, k, phi, w)
    }
    
    func predict(ptIni: Tensor<Float>, seqStr inputSeqStr: Tensor<Int32>, seqStrMask: Tensor<Float>, bias: Float = 0) -> (Tensor<Float>, Tensor<Float>) {
        let bias = Tensor<Float>(bias)
        let batchSize = ptIni.shape[0]
        let kIni = Tensor<Float>(zeros: [batchSize, 10])
        let wIni = Tensor<Float>(zeros: [batchSize, 81])
        
        var seqStr = Tensor<Float>(zeros: [inputSeqStr.shape[0], inputSeqStr.shape[1], 81])
        seqStr.scatter(indices: inputSeqStr, scalar: Tensor(1))
        
        var mask = Tensor<Float>(ones: [batchSize])
        var seqPt: [Tensor<Float>] = [ptIni]
        var seqMask: [Tensor<Float>] = [mask]
        
        let lastChar = Int((seqStrMask.sum(alongAxes: 0) - 1).scalars[0])
        
        var pt = ptIni
        var hState = PyTorchLSTMCell.State(hidden: .init(zeros: [batchSize, 900]), cell: .init(zeros: [batchSize, 900]))
        var k = kIni
        var w = wIni
        for _ in 0..<10000 {
            let result = rnnStep(inputs: pt, hStatePre: hState, kPre: k, wPre: w, c: seqStr, cMask: seqStrMask, mask: mask)
            hState = result.0
            let h = hState.hidden
            k = result.1
            let phi = result.2
            w = result.3
            let hw = h.concatenated(with: w, alongAxis: -1)
            pt = mixture(.init(h: hw, bias: bias))
            seqPt.append(pt)
            
            let lastPhi = phi[lastChar][0].scalar!
            let maxPhi = phi.flattened().max().scalar!
            mask = mask * Tensor(Float(lastPhi >= (0.95 * maxPhi) ? 0 : 1))
            seqMask.append(mask)
            if mask.sum() == Tensor(0) {
                break
            }
        }
        
        return (Tensor(seqPt), Tensor(seqMask))
    }
}

func visualize(strokes x: Tensor<Float>) {
    let mean = Tensor<Float>([[8.1842, 0.1145]])
    let std = Tensor<Float>([[40.3660, 37.0441]])
    
    var stroke = (x.slice(lowerBounds: [0, 0], upperBounds: [-1, 2]) * std + mean).cumulativeSum(alongAxis: 0)
    stroke = stroke * Tensor<Float>(ones: [x.shape[0], 1]).concatenated(with: -Tensor<Float>(ones: [x.shape[0], 1]), alongAxis: -1)
    let pen = x.slice(lowerBounds: [0, 2], upperBounds: [-1, 3]).reshaped(to: [-1])
    
    let min = stroke.min(alongAxes: 0)[0]
    let (xmin, ymin) = (min[0].scalar!, min[1].scalar!)
    let max = stroke.max(alongAxes: 0)[0]
    let (xmax, ymax) = (max[0].scalar!, max[1].scalar!)
    
    var actions: [PythonObject] = [matplotlib.path.Path.MOVETO]
    var coords: [[Int]] = []
    for i in 0..<x.shape[0] {
        let (c, p) = (stroke[i], pen[i].scalar!)
        if p >= -0.0001 {
            coords.append([Int(c[0].scalar!), Int(c[1].scalar!)])
            if p == 1 {
                actions.append(matplotlib.path.Path.MOVETO)
            } else {
                actions.append(matplotlib.path.Path.LINETO)
            }
        }
    }
    actions.removeLast()
    let ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    let path = matplotlib.path.Path(coords, actions)
    let patch = matplotlib.patches.PathPatch(path, facecolor: "none")
    ax.add_patch(patch)
    
    plt.show()
}

let charDict = Dictionary<Character, Int>(uniqueKeysWithValues: " !\"#%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz".enumerated().map { ($0.element, $0.offset) })

func loadModel() -> HandwritingGenerator {
    var m = HandwritingGenerator()
    let weights = h5py.File("graves_handwriting_generation_2018-03-13-02-45epoch_49.hd5", "r")
    m.attention.linear.weight = Tensor<Float>(numpy: weights["attention.linear.weight"].value.astype("float32"))!.transposed()
    m.attention.linear.bias = Tensor<Float>(numpy: weights["attention.linear.bias"].value.astype("float32"))!
    m.mixture.linear.weight = Tensor<Float>(numpy: weights["mixture.linear.weight"].value.astype("float32"))!.transposed()
    m.mixture.linear.bias = Tensor<Float>(numpy: weights["mixture.linear.bias"].value.astype("float32"))!
    m.rnnCell.ih.weight = Tensor<Float>(numpy: weights["rnn_cell.cell.weight_ih"].value.astype("float32"))!.transposed()
    m.rnnCell.ih.bias = Tensor<Float>(numpy: weights["rnn_cell.cell.bias_ih"].value.astype("float32"))!
    m.rnnCell.hh.weight = Tensor<Float>(numpy: weights["rnn_cell.cell.weight_hh"].value.astype("float32"))!.transposed()
    m.rnnCell.hh.bias = Tensor<Float>(numpy: weights["rnn_cell.cell.bias_hh"].value.astype("float32"))!
    return m
}

func generateAndVisualize(m: HandwritingGenerator, text: String, bias: Float) {
    let iSeqPt = Tensor<Float>([[0, 0, 1]])
    let iSeqStr = Tensor("\(text) ".map { Int32(charDict[$0]!) }).reshaped(to: [-1, 1])
    let iSeqStrMask = Tensor<Float>(ones: iSeqStr.shape)
    let (seqPt, _) = m.predict(ptIni: iSeqPt, seqStr: iSeqStr, seqStrMask: iSeqStrMask, bias: bias)

    plt.figure(figsize: [10, 2])
    visualize(strokes: seqPt.squeezingShape(at: 1))
}

generateAndVisualize(m: loadModel(), text: CommandLine.arguments[1], bias: Float(CommandLine.arguments[2])!)
