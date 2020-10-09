// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import TensorFlow

// MARK: Solution of shallow water equation

/// Differentiable solution of shallow water equation on a unit square.
///
/// Shallow water equation is a type of hyperbolic partial differential equation (PDE). This struct
/// represents its solution calculated with finite-difference discretization on a 2D plane and at a
/// particular point in time.
///
/// More details about the shallow water PDE can found for example on
/// [Wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations)
///
/// # Domain and Discretization
/// The PDE is solved on a `<0,1>x<0,1>` square discretized with spatial step of size `Δx`.
/// Laplace operator is approximated with five-point stencil finite-differencing.
///
/// Temporal advancing uses semi implicit Euler's schema. Time step `Δt` is calculated from
/// `Δx` to stay below the Courant–Friedrichs–Lewy numerical stability limit.
///
/// # Boundary Conditions
/// Values around the edges of the domain are subject to trivial Dirichlet boundary conditions
/// (i.e. equal to 0 with an arbitrary gradient).
///
/// # Laplace Operator Δ
/// Discretization of the operator is implemented as operations on shifted slices of the water
/// height field. Especially in TensorFlow eager mode this provides better performance because
/// there's a small number of dispatched operations that operate on larger input tensors.
///
/// - Bug:
///  Result of applying Laplacian to the water height at a particular time-step is needlessly
///  (due to my laziness ;) calculated twice. When the time advances `u1` becomes `u0` of
///  the following time step. The result of applying the Laplace operator to `u1` can be cached
///  and reused a step later.
///
struct TensorSliceSolution: ShallowWaterEquationSolution {
  /// Water level height
  var waterLevel: [[Float]] { u1.array.map { $0.scalars } }
  /// Solution time
  var time: Float { t }

  /// Height of the water surface at time `t`
  private var u1: Tensor<Float>
  /// Height of the water surface at previous time-step `t - Δt`
  private var u0: Tensor<Float>
  /// Solution time
  @noDerivative private let t: Float
  /// Speed of sound
  @noDerivative private let c: Float = 340.0
  /// Dispersion coefficient
  @noDerivative private let α: Float = 0.00001
  /// Number of spatial grid points
  @noDerivative private let resolution: Int = 256
  /// Spatial discretization step
  @noDerivative private var Δx: Float { 1 / Float(resolution) }
  /// Time-step calculated to stay below the CFL stability limit
  @noDerivative private var Δt: Float { (sqrt(α * α + Δx * Δx / 3) - α) / c }

  /// Creates initial solution with water level `u0` at time `t`.
  @differentiable
  init(waterLevel u0: Tensor<Float>, time t: Float = 0.0) {
    self.u0 = u0
    self.u1 = u0
    self.t = t

    assert(u0.shape.count == 2)
    assert(u0.shape[0] == resolution && u0.shape[1] == resolution)
  }

  /// Calculates solution stepped forward by one time-step `Δt`.
  ///
  /// - `u0` - Water surface height at previous time step
  /// - `u1` - Water surface height at current time step
  /// - `u2` - Water surface height at next time step (calculated)
  @differentiable
  func evolved() -> TensorSliceSolution {
    var Δu0 = Δ(u0)
    var Δu1 = Δ(u1)
    Δu0 = Δu0.padded(
      forSizes: [
        (before: 1, after: 1),
        (before: 1, after: 1),
      ], with: 0.0)
    Δu1 = Δu1.padded(
      forSizes: [
        (before: 1, after: 1),
        (before: 1, after: 1),
      ], with: 0.0)

    let Δu0Coefficient = c * α * Δt
    let Δu1Coefficient = c * c * Δt * Δt + c * α * Δt
    let cΔu0 = Δu0Coefficient * Δu0
    let cΔu1 = Δu1Coefficient * Δu1

    let u1twice = 2.0 * u1
    let u2 = u1twice + cΔu1 - u0 - cΔu0

    LazyTensorBarrier(wait: true)
    return TensorSliceSolution(u0: u1, u1: u2, t: t + Δt)
  }

  /// Constructs intermediate solution with previous water level `u0`, current water level `u1` and time `t`.
  @differentiable
  private init(u0: Tensor<Float>, u1: Tensor<Float>, t: Float) {
    self.u0 = u0
    self.u1 = u1
    self.t = t

    assert(u0.shape.count == 2)
    assert(u0.shape[0] == resolution && u0.shape[1] == resolution)
    assert(u1.shape.count == 2)
    assert(u1.shape[0] == resolution && u1.shape[1] == resolution)
  }

  /// Applies discretized Laplace operator to scalar field `u`.
  @differentiable
  private func Δ(_ u: Tensor<Float>) -> Tensor<Float> {
    assert(u.shape.allSatisfy { $0 > 2 })
    assert(u.rank == 2)

    let sliceShape = Tensor(copying: (u.shape - 2).tensor, to: u.device)

    let left = u.slice(lowerBounds: Tensor([0, 1], on: u.device), sizes: sliceShape)
    let right = u.slice(lowerBounds: Tensor([2, 1], on: u.device), sizes: sliceShape)
    let up = u.slice(lowerBounds: Tensor([1, 0], on: u.device), sizes: sliceShape)
    let down = u.slice(lowerBounds: Tensor([1, 2], on: u.device), sizes: sliceShape)
    let center = u.slice(lowerBounds: Tensor([1, 1], on: u.device), sizes: sliceShape)

    let center4 = center * 4.0
    let finiteDifference = left + right + up + down - center4
    let Δu = finiteDifference / Δx / Δx

    return Δu
  }
}

// MARK: - Cost calculated as mean L2 distance to a target image

extension TensorSliceSolution {
  /// Calculates mean squared error loss between the solution and a `target` grayscale image.
  @differentiable
  func meanSquaredError(to target: Tensor<Float>) -> Float {
    assert(target.shape.count == 2)
    assert(target.shape[0] == resolution && target.shape[1] == resolution)

    let error = u1 - target
    return error.squared().mean().scalarized()
  }
}

// MARK: - Utilities

extension TensorShape {
  fileprivate var tensor: Tensor<Int32> { Tensor<Int32>(dimensions.map(Int32.init)) }

  fileprivate static func - (lhs: TensorShape, rhs: Int) -> TensorShape {
    TensorShape(lhs.dimensions.map { $0 - rhs })
  }
}
