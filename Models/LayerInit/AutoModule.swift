import TensorFlow

public protocol AutoModule: AutoLayer {
    associatedtype LayerType: AutoLayer

    var initializeLayer: LayerType { mutating get }
}

extension AutoModule  {
    public typealias InstanceType = LayerType.InstanceType
    public typealias InputShape = LayerType.InputShape
    public typealias OutputShape = LayerType.OutputShape

    public func buildModelWithOutputShape(inputShape: InputShape) -> (InstanceType, OutputShape) {
        var selfCopy = self
        return selfCopy.initializeLayer.buildModelWithOutputShape(inputShape: inputShape)
    }
}
