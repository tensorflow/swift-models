import TensorFlow

public protocol AutoModule: AutoLayer {
    associatedtype LayerType: AutoLayer

    var initializeLayer: LayerType { mutating get }
}

extension AutoModule  {
    public typealias InstanceType = LayerType.InstanceType
    public typealias InputShape = LayerType.InputShape
    public typealias OutputShape = LayerType.OutputShape

    public func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape) {
        var selfCopy = self
        return selfCopy.initializeLayer.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: keyPathSoFar, keyDict: &keyDict)
    }
}
