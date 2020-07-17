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

public class FunctionalLayer : Hashable {
    func outputShape() -> [Int] {
        fatalError("Must implement")
    }

    func getLayer() -> DynamicLayerStore {
        fatalError("Must implement")
    }

    func getDependencies() -> [FunctionalLayer] {
        fatalError("Must implement")
    }

    func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        fatalError("Must implement")
    }

    public static func ==(lhs: FunctionalLayer, rhs: FunctionalLayer) -> Bool {
        return lhs === rhs
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

extension FunctionalLayer {
    public func build() -> ComposedLayer {
        // compute topological sort
        var allLayers: [FunctionalLayer] = []
        var toVisit: [FunctionalLayer] = [self] // TODO(shadaj): should be a queue
        var unresolvedDependenciesPerLayer: [FunctionalLayer:Int] = [:]
        var allDependenciesMet: [FunctionalLayer] = []
        
        var dependents: [FunctionalLayer:[FunctionalLayer]] = [:]
        var dependentsCount: [FunctionalLayer : Int] = [:]

        while toVisit.count > 0 {
            let next = toVisit.removeFirst()
            if (!allLayers.contains(next)) {
                allLayers.append(next)
                let dependencies = next.getDependencies()
                
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
                    allDependenciesMet.append(next)
                }
            }
        }

        var allLayersBuilt: [DynamicLayerStore] = []
        for layer in allLayers {
            allLayersBuilt.append(layer.getLayer())
        }

        var layerComputeOrder: [FunctionalLayer] = []
        var layersBuilt: [DynamicLayerStore] = []
        var layerToIndex: [FunctionalLayer : Int] = [:]

        while allDependenciesMet.count > 0 {
            let next = allDependenciesMet.removeFirst()
            layerComputeOrder.append(next)
            layersBuilt.append(next.getLayer())
            layerToIndex[next] = layersBuilt.count - 1
            for dependent in dependents[next, default: []] {
                unresolvedDependenciesPerLayer[dependent]! -= 1
                if unresolvedDependenciesPerLayer[dependent]! == 0 {
                    allDependenciesMet.append(dependent)
                }
            }
        }

        var accumulatedFunction: @differentiable (inout [Tensor<Float>], [DynamicLayerStore]) -> Void = {_,_ in}

        var lastIndex = 0
        var maxIndex = 0
        var allocatedIndices: [FunctionalLayer : Int] = [:]
        var openSlots: [Int] = []

        for (layerIndex, layer) in layerComputeOrder.enumerated() {
            var dependencyIndices: [Int] = []
            for dependency in layer.getDependencies() {
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
