import TensorFlow

public typealias Tensorf = Tensor<Float>

#if os(macOS)
func random() -> UInt32 {
    arc4random()
}
#endif
