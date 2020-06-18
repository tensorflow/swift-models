import TensorFlow

struct HandwritingGenerator {
    static let charDict = Dictionary<Character, Int>(
        uniqueKeysWithValues: " !\"#%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
            .enumerated().map { ($0.element, $0.offset) })

    var attention: Attention
    var rnnCell: PyTorchLSTMCell
    var mixture: MixtureGaussians2DAndPen
    
    init() {
        attention = Attention(inputs: 900, mixtureComponents: 10)
        rnnCell = PyTorchLSTMCell(inputSize: 81, hiddenSize: 900)
        mixture = MixtureGaussians2DAndPen(inputs: 900 + 81, mixtureComponents: 20)
    }
    
    func rnnStep(inputs: Tensor<Float>,
                 hStatePre: PyTorchLSTMCell.State,
                 kPre: Tensor<Float>,
                 wPre: Tensor<Float>,
                 c: Tensor<Float>,
                 cMask: Tensor<Float>,
                 mask: Tensor<Float>
                 ) -> (PyTorchLSTMCell.State, Tensor<Float>, Tensor<Float>, Tensor<Float>) {
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
    
    func predict(ptIni: Tensor<Float>,
                 seqStr inputSeqStr: Tensor<Int32>,
                 seqStrMask: Tensor<Float>,
                 bias: Float = 0) -> (Tensor<Float>, Tensor<Float>) {
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
        var hState = PyTorchLSTMCell.State(hidden: .init(zeros: [batchSize, 900]),
                                           cell: .init(zeros: [batchSize, 900]))
        var k = kIni
        var w = wIni
        for _ in 0..<10000 {
            let result = rnnStep(inputs: pt, hStatePre: hState, kPre: k,
                                 wPre: w, c: seqStr, cMask: seqStrMask, mask: mask)
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
