import Foundation
import TensorFlow
import FastStyleTransfer

func printUsage() {
    let exec = URL(string: CommandLine.arguments[0])!.lastPathComponent
    print("Usage:")
    print("\(exec) --weights=<path> --image=<path> --output=<path>")
    print("    --weights: Path to weights in TF checkpoint V2 format")
    print("    --image: Path to image in JPEG format")
    print("    --output: Path to output image")
}

/// Startup parameters.
struct Config {
    var weights: String? = "FastStyleTransfer/Demo/weights/candy"
    var image: String? = nil
    var output: String? = "out.jpg"
}

var config = Config()
parseArguments(
    into: &config,
    with: [
        "weights": \Config.weights,
        "image": \Config.image,
        "output": \Config.output
    ]
)

guard let image = config.image, let output = config.output else {
    print("Error: No input image!")
    printUsage()
    exit(1)
}

guard let imageTensor = try? loadJpegAsTensor(from: image) else {
    print("Error: Failed to load image \(image). Check file exists and has JPEG format")
    printUsage()
    exit(1)
}

// Init the model.
var style = TransformerNet()
do {
    try importWeights(&style, from: config.weights!)
} catch {
    print("Error: Failed to load weights \(config.weights!). Check path exists and contains TF checkpoint")
    printUsage()
    exit(1)
}

// Apply the model to image.
let out = style(imageTensor.expandingShape(at: 0))

saveTensorAsJpeg(out.squeezingShape(at: 0), to: output)
print("Written output to \(output)")
