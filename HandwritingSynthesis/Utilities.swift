import TensorFlow
import PythonKit

let plt = Python.import("matplotlib.pyplot")
let matplotlib = Python.import("matplotlib")

func visualize(strokes x: Tensor<Float>) {
    let mean = Tensor<Float>([[8.1842, 0.1145]])
    let std = Tensor<Float>([[40.3660, 37.0441]])
    
    var stroke = (x.slice(lowerBounds: [0, 0], upperBounds: [-1, 2]) * std + mean)
        .cumulativeSum(alongAxis: 0)
    stroke *= Tensor<Float>(ones: [x.shape[0], 1])
        .concatenated(with: -Tensor<Float>(ones: [x.shape[0], 1]), alongAxis: -1)
    let pen = x.slice(lowerBounds: [0, 2], upperBounds: [-1, 3]).reshaped(to: [-1])
    
    let min = stroke.min(alongAxes: 0)[0]
    let (xmin, ymin) = (min[0].scalar!, min[1].scalar!)
    let max = stroke.max(alongAxes: 0)[0]
    let (xmax, ymax) = (max[0].scalar!, max[1].scalar!)
    
    var actions: [PythonObject] = [matplotlib.path.Path.MOVETO]
    var coords: [[Int]] = []
    for i in 0..<x.shape[0] {
        let (c, p) = (stroke[i], pen[i].scalar!)
        if p >= -0.0001 {
            coords.append([Int(c[0].scalar!), Int(c[1].scalar!)])
            if p == 1 {
                actions.append(matplotlib.path.Path.MOVETO)
            } else {
                actions.append(matplotlib.path.Path.LINETO)
            }
        }
    }
    actions.removeLast()
    let ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    let path = matplotlib.path.Path(coords, actions)
    let patch = matplotlib.patches.PathPatch(path, facecolor: "none")
    ax.add_patch(patch)
    
    plt.show()
}

func multinomial(_ x: [Float]) -> Int {
    let y = Float.random(in: 0...1)
    var z: Float = 0
    for (iIdx, i) in x.enumerated() {
        z += i
        if z >= y {
            return iIdx
        }
    }
    fatalError()
}

extension Tensor {
    mutating func scatter(indices: Tensor<Int32>, scalar: Tensor<Scalar>) {
        for i in 0..<shape[0] {
            for j in 0..<shape[1] {
                self[i][j][Int(indices[i][j].scalar!)] = scalar
            }
        }
    }
}
