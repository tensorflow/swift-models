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

#if os(Windows)
#if canImport(CRT)
import CRT
#else
import MSVCRT
#endif
#endif

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
/// Discretization of the operator is implemented as tight loops over the elements of the water
/// height tensor. This is very unlikely to give good performance because Tensors are not
/// optimized for element-wise access. Especially in eager mode there's a significant overhead
/// for every method called on a Tensor.
///
struct TensorLoopSolution: ShallowWaterEquationSolution {
  /// Water level height
  var waterLevel: Tensor<Float> { u1 }
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
  @noDerivative private let resolution: Int
  /// Spatial discretization step
  @noDerivative private var Δx: Float { 1 / Float(resolution) }
  /// Time-step calculated to stay below the CFL stability limit
  @noDerivative private var Δt: Float { (sqrt(α * α + Δx * Δx / 3) - α) / c }

  /// Creates initial solution with water level `u0` at time `t` using the specified TensorFlow `device`.
  @differentiable
  init(waterLevel u0: Tensor<Float>, time t: Float = 0.0) {
    self.resolution = u0.shape[0]
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
  func evolved() -> TensorLoopSolution {
    var u2 = u1

    for x in withoutDerivative(at: 1..<resolution - 1) {
      for y in withoutDerivative(at: 1..<resolution - 1) {
        let Δu0 = Δ(u0, x, y)
        let Δu1 = Δ(u1, x, y)

        let Δu0Coefficient = c * α * Δt
        let Δu1Coefficient = c * c * Δt * Δt + c * α * Δt

        let cΔu0 = Δu0Coefficient * Δu0
        let cΔu1 = Δu1Coefficient * Δu1

        let u12 = 2.0 * u1[x, y]
        let u01 = u0[x, y]

        let result = u12 + cΔu1 - u01 - cΔu0
        // FIXME: Should be u2[x][y] = result
        u2.update(x, y, to: result)
      }
    }

    LazyTensorBarrier(wait: true)
    return TensorLoopSolution(u0: u1, u1: u2, t: t + Δt)
  }

  /// Constructs intermediate solution with previous water level `u0`, current water level `u1` and time `t`.
  @differentiable
  private init(u0: Tensor<Float>, u1: Tensor<Float>, t: Float) {
    self.resolution = u0.shape[0]
    self.u0 = u0
    self.u1 = u1
    self.t = t

    assert(u0.shape.count == 2)
    assert(u0.shape[0] == resolution && u0.shape[1] == resolution)
    assert(u1.shape.count == 2)
    assert(u1.shape[0] == resolution && u1.shape[1] == resolution)
  }

  /// Applies discretized Laplace operator to scalar field `u` at grid points `x` and `y`.
  @differentiable
  private func Δ(_ u: Tensor<Float>, _ x: Int, _ y: Int) -> Tensor<Float> {
    let left = u[x - 1, y]
    let right = u[x + 1, y]
    let up = u[x, y + 1]
    let down = u[x, y - 1]
    let center = u[x, y]

    let center4 = center * 4.0
    let finiteDifference = left + right + up + down - center4
    let Δu = finiteDifference / Δx / Δx
    return Δu
  }
}

// MARK: - Cost calculated as mean L2 distance to a target image

extension TensorLoopSolution {

  /// Calculates mean squared error loss between the solution and a `target` grayscale image.
  @differentiable
  func meanSquaredError(to target: Tensor<Float>) -> Float {
    assert(target.shape.count == 2)
    assert(target.shape[0] == resolution && target.shape[1] == resolution)

    let error = u1 - target
    return error.squared().mean().scalarized()
  }
}

// MARK: - Workaround for non-differentiable coroutines
// https://bugs.swift.org/browse/TF-1078
// https://bugs.swift.org/browse/TF-1080

extension Tensor where Scalar: TensorFlowFloatingPoint {

  @differentiable(wrt: (self, value))
  fileprivate mutating func update(_ x: Int, _ y: Int, to value: Self) {
    assert(value.shape.contiguousSize == 1)
    self[x, y] = value
  }

  @derivative(of: update, wrt: (self, value))
  fileprivate mutating func vjpUpdate(_ x: Int, _ y: Int, to value: Self) -> (
    value: (), pullback: (inout Self) -> Self
  ) {
    self.update(x, y, to: value)

    func pullback(`self`: inout Self) -> Self {
      let `value` = `self`[x, y]
      `self`[x, y] = Tensor(0)
      return `value`
    }
    return ((), pullback)
  }
}
