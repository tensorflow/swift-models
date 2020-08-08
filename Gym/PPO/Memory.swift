class Memory {
    var states: [[Float]] = []
    var actions: [Int32] = []
    var rewards: [Float] = []
    var logProbs: [Float] = []
    var isDones: [Bool] = []

    init() {}

    func clear_memory() {
        states.removeAll()
        actions.removeAll()
        rewards.removeAll()
        logProbs.removeAll()
        isDones.removeAll()
    }
}
