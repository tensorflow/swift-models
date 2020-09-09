import TensorFlow


// MARK: Solution of shallow water equation

/// Differentiable solution of shallow water equation on a unit square.
protocol ShallowWaterEquationSolution: Differentiable {    
    /// Snapshot of water level height at time `time`.
    @noDerivative var waterLevel: [[Float]] { get }
    /// Solution time
    @noDerivative var time: Float { get }

    /// Returns solution evolved forward in time by one step.
    @differentiable
    func evolved() -> Self
}


// MARK: - Evolution of the solution in time

extension Array where Array.Element: ShallowWaterEquationSolution {

    /// Creates an array of shallow water equation solutions by evolving the `initialSolution` forward `numSteps`-times.
    @differentiable
    init(evolve initialSolution: Array.Element, for numSteps: Int) {
        self.init()

        var currentSolution = initialSolution
        for _ in 0 ..< numSteps {
            self.append(currentSolution)
            currentSolution = currentSolution.evolved()
        }
        self.append(currentSolution)
    }
}
