import TensorFlow
import ModelSupport


// MARK: Visualization of shallow water equation solution

/// Visualization of the solution at a particular time-step.
struct SolutionVisualization<Solution: ShallowWaterEquationSolution> {
    let solution: Solution

    /// Returns a top-down mosaic of the water level colored by its height.
    var waterLevel: Image {
        let square = TensorShape([solution.waterLevel.count, solution.waterLevel.count])
        let waterLevel = Tensor(shape: square, scalars: solution.waterLevel.flatMap { $0 })
        let normalizedWaterLevel = waterLevel.normalized(min: -1, max: +1)
        return Image(tensor: normalizedWaterLevel)
    }
}

extension ShallowWaterEquationSolution {
    var visualization: SolutionVisualization<Self> { SolutionVisualization(solution: self) }
}


// MARK: - Utilities

fileprivate extension Tensor where Scalar == Float {
    /// Returns image normalized from `min`-`max` range to standard 0-255 range and converted to `UInt8`.
    func normalized(min: Scalar = -1, max: Scalar = +1) -> Tensor<UInt8> {
        precondition(max > min)

        let clipped = self.clipped(min: min, max: max)
        let normalized = (clipped - min) / (max - min) * Float(UInt8.max)
        return Tensor<UInt8>(normalized)
    }
}
