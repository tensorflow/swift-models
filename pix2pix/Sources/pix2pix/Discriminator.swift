import TensorFlow

public struct NetD: Layer {
    var module: Sequential<Sequential<Conv2D<Float>, Sequential<LeakyRELU, Sequential<Conv2D<Float>, Sequential<BatchNorm<Float>, Sequential<LeakyRELU, Sequential<Conv2D<Float>, Sequential<BatchNorm<Float>, LeakyRELU>>>>>>>, Sequential<ConvLayer, Sequential<BatchNorm<Float>, Sequential<LeakyRELU, ConvLayer>>>>
    
    public init(inChannels: Int, lastConvFilters: Int) {
        let kw = 4
        
        let module = Sequential {
            Conv2D<Float>(filterShape: (kw, kw, inChannels, lastConvFilters),
                          strides: (2, 2),
                          padding: .same,
                          filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
            LeakyRELU()
            
            Conv2D<Float>(filterShape: (kw, kw, lastConvFilters, 2 * lastConvFilters),
                          strides: (2, 2),
                          padding: .same,
                          filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
            BatchNorm<Float>(featureCount: 2 * lastConvFilters)
            LeakyRELU()
            
            Conv2D<Float>(filterShape: (kw, kw, 2 * lastConvFilters, 4 * lastConvFilters),
                          strides: (2, 2),
                          padding: .same,
                          filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
            BatchNorm<Float>(featureCount: 4 * lastConvFilters)
            LeakyRELU()
        }
        
        let module2 = Sequential {
            module
            ConvLayer(inChannels: 4 * lastConvFilters, outChannels: 8 * lastConvFilters,
                      kernelSize: 4, stride: 1, padding: 1)
            
            BatchNorm<Float>(featureCount: 8 * lastConvFilters)
            LeakyRELU()
            
            ConvLayer(inChannels: 8 * lastConvFilters, outChannels: 1,
                      kernelSize: 4, stride: 1, padding: 1)
        }
        
        self.module = module2
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        return self.module(input)
    }
}
