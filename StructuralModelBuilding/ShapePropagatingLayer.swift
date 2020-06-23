import PenguinStructures

// TODO: Pick a better name.
protocol ShapePropagatingLayer: SequentialLayer {
    associatedtype InputShape: FixedSizeArray
    associatedtype OutputShape: FixedSizeArray
    associatedtype HParams
}
