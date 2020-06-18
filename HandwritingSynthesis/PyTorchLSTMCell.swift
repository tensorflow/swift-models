import TensorFlow

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
        var forgetGate = gates.slice(lowerBounds: [0, hiddenSize],
                                     upperBounds: [batchSize, hiddenSize * 2])
        var cellGate = gates.slice(lowerBounds: [0, hiddenSize * 2],
                                   upperBounds: [batchSize, hiddenSize * 3])
        var outGate = gates.slice(lowerBounds: [0, hiddenSize * 3],
                                  upperBounds: [batchSize, hiddenSize * 4])
        
        inGate = sigmoid(inGate)
        forgetGate = sigmoid(forgetGate)
        cellGate = tanh(cellGate)
        outGate = sigmoid(outGate)
        
        let cy = (forgetGate * cx) + (inGate * cellGate)
        let hy = outGate * tanh(cy)
        
        return .init(hidden: hy, cell: cy)
    }
}
