import TensorFlow

// workaround for https://bugs.swift.org/browse/TF-1078
public extension Array {
  /// A functional version of `Array.subscript.modify`.
  /// Differentiation does yet not support `Array.subscript.modify` because
  /// it is a coroutine.
  @differentiable(where Element: Differentiable)
  mutating func updated(at index: Int, with newValue: Element) {
    self[index] = newValue
  }
}

public extension Array where Element: Differentiable {
  @derivative(of: updated)
  mutating func vjpUpdated(at index: Int, with newValue: Element)
    -> (value: Void, pullback: (inout TangentVector) -> (Element.TangentVector)) {
    self.updated(at: index, with: newValue)
    return ((), { v in
       let dElement = v[index]
      if index < v.base.count{
          v.base[index] = .zero
      }
      return dElement
    })
  }
}

public class AnyTracingLayer: Hashable {
    /// The shape of the tensor emitted by this layer
    var outputShape: [Int] {
        fatalError("Must implement")
    }

    /// Constructs the underlying classic layer wrapped into a type-erased container
    func makeClassicLayer() -> DynamicLayerStore {
        fatalError("Must implement")
    }

    /// Gets the list of immediate dependencies of the current layer, whose outputs are used in the current layer's computation
    var dependencies: [AnyTracingLayer] {
        fatalError("Must implement")
    }

    /// Returns a closure which executes the layer by pulling inputs from a dependency source and calling the classic layer
    /// - dependencyIndices: the indices of the cache at which the layer's dependencies lie; each index in the array corresponds
    ///   to the layer at the same index in getDependencies()
    func buildLayerApplication(dependencyIndices: [Int]) // TODO(shadaj): layerApplication
        -> @differentiable (_ dependencySource: [Tensor<Float>], _ classicLayer: DynamicLayerStore) -> Tensor<Float> {
        fatalError("Must implement")
    }

    public static func ==(lhs: AnyTracingLayer, rhs: AnyTracingLayer) -> Bool {
        return lhs === rhs
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

/// A specification for a layer and all its dependencies.
public class TracingLayer<T> : AnyTracingLayer {
    
}

extension AnyTracingLayer {
    /// Constructs an instance of the layer graph specified by `self`.
    public func build() -> ComposedLayer {
        // first, explore the graph to locate all layers and precompute values for topological sort
        var allLayers: [AnyTracingLayer] = []
        var toVisit: [AnyTracingLayer] = [self] // TODO(shadaj): should be a queue
        var unresolvedDependenciesPerLayer: [AnyTracingLayer:Int] = [:]
        var inputLayer: AnyTracingLayer? = nil
        
        var dependents: [AnyTracingLayer: [AnyTracingLayer]] = [:]
        var dependentsCount: [AnyTracingLayer : Int] = [:]

        while toVisit.count > 0 {
            let next = toVisit.removeFirst()
            if (!allLayers.contains(next)) {
                allLayers.append(next)
                
                let dependencies = next.dependencies
                unresolvedDependenciesPerLayer[next] = dependencies.count
                
                if dependencies.count > 0 {
                    for dependency in dependencies {
                        if dependents[dependency] == nil {
                            dependents[dependency] = []
                            dependentsCount[dependency] = 0
                        }

                        dependents[dependency]!.append(next)
                        dependentsCount[dependency]! += 1
                        toVisit.append(dependency)
                    }
                } else {
                    // we've found the input layer, which has no dependencies
                    inputLayer = next
                }
            }
        }

        // compute topological sort
        var allDependenciesMet: [AnyTracingLayer] = [inputLayer!]
        var layerComputeOrder: [AnyTracingLayer] = []
        var layersBuilt: [DynamicLayerStore] = []
        var layerToIndex: [AnyTracingLayer : Int] = [:]

        while allDependenciesMet.count > 0 {
            let next = allDependenciesMet.removeFirst()
            layerComputeOrder.append(next)
            layersBuilt.append(next.makeClassicLayer())
            layerToIndex[next] = layersBuilt.count - 1
            for dependent in dependents[next, default: []] {
                unresolvedDependenciesPerLayer[dependent]! -= 1
                if unresolvedDependenciesPerLayer[dependent]! == 0 {
                    allDependenciesMet.append(dependent)
                }
            }
        }

        // build out the function that executes all layers in the order determined by the topological sort
        var accumulatedFunction: @differentiable (inout [Tensor<Float>], [DynamicLayerStore]) -> Void = {_,_ in}

        var lastIndex = 0
        var maxIndex = 0
        var allocatedIndices: [AnyTracingLayer : Int] = [:]
        var openSlots: [Int] = []

        for (layerIndex, layer) in layerComputeOrder.enumerated() {
            var dependencyIndices: [Int] = []
            for dependency in layer.dependencies {
                let previouslyAllocated = allocatedIndices[dependency]!
                dependencyIndices.append(previouslyAllocated)
                
                dependentsCount[dependency]! -= 1
                if dependentsCount[dependency] == 0 {
                    // we read the dependencies before we write, so it's safe to output to a slot of a dependency
                    openSlots.append(previouslyAllocated)
                }
            }
            
            if dependencyIndices.count == 0 { // input layer
                dependencyIndices.append(0)
                openSlots.append(0) // the input value is only used once, in the single input layer
            }

            let layerCaller = layer.buildLayerApplication(dependencyIndices: dependencyIndices)

            let prevAccumulated = accumulatedFunction
            let allocatedSlot = openSlots.count > 0 ? openSlots.removeFirst() : maxIndex + 1

            if allocatedSlot > maxIndex {
                assert(allocatedSlot == maxIndex + 1)
                accumulatedFunction = { (outputs: inout [Tensor<Float>], layers: [DynamicLayerStore]) in
                    prevAccumulated(&outputs, layers)
                    outputs.append(layerCaller(outputs, layers[layerIndex]))
                }
            } else {
                accumulatedFunction = { (outputs: inout [Tensor<Float>], layers: [DynamicLayerStore]) in
                    prevAccumulated(&outputs, layers)
                    outputs.updated(at: allocatedSlot, with: layerCaller(outputs, layers[layerIndex]))
                    // faster: outputs = [underlyingFunction(outputs, layers[index])]
                }
            }

            allocatedIndices[layer] = allocatedSlot
            lastIndex = allocatedSlot
            maxIndex = max(maxIndex, allocatedSlot)
        }

        return ComposedLayer(
            layers: layersBuilt,
            callFunction: { (layers: [DynamicLayerStore], input: Tensor<Float>) in
                var outputs: [Tensor<Float>] = [input]
                accumulatedFunction(&outputs, layers)
                return outputs[lastIndex]
            }
        )
    }
}
