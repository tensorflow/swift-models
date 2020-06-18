import TensorFlow
import PythonKit

let os = Python.import("os")
let h5py = Python.import("h5py")

func loadModel() -> HandwritingGenerator {
    var m = HandwritingGenerator()
    if !Bool(os.path.isfile("graves_handwriting_generation_2018-03-13-02-45epoch_49.hd5"))! {
        fatalError("ERROR: You must run download_model.sh before attempting to generate images.")
    }
    let weights = h5py.File("graves_handwriting_generation_2018-03-13-02-45epoch_49.hd5", "r")
    m.attention.linear.weight = Tensor<Float>(
        numpy: weights["attention.linear.weight"].value.astype("float32"))!.transposed()
    m.attention.linear.bias = Tensor<Float>(
        numpy: weights["attention.linear.bias"].value.astype("float32"))!
    m.mixture.linear.weight = Tensor<Float>(
        numpy: weights["mixture.linear.weight"].value.astype("float32"))!.transposed()
    m.mixture.linear.bias = Tensor<Float>(
        numpy: weights["mixture.linear.bias"].value.astype("float32"))!
    m.rnnCell.ih.weight = Tensor<Float>(
        numpy: weights["rnn_cell.cell.weight_ih"].value.astype("float32"))!.transposed()
    m.rnnCell.ih.bias = Tensor<Float>(
        numpy: weights["rnn_cell.cell.bias_ih"].value.astype("float32"))!
    m.rnnCell.hh.weight = Tensor<Float>(
        numpy: weights["rnn_cell.cell.weight_hh"].value.astype("float32"))!.transposed()
    m.rnnCell.hh.bias = Tensor<Float>(
        numpy: weights["rnn_cell.cell.bias_hh"].value.astype("float32"))!
    return m
}
