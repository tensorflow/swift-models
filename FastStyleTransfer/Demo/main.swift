import FastStyleTransfer
import Foundation
import ModelSupport
import TensorFlow

func printUsage() {
    let exec = URL(string: CommandLine.arguments[0])!.lastPathComponent
    print("Usage:")
    print("\(exec) --style=<name> --image=<path> --output=<path>")
    print("    --style: Style to use (candy, mosaic, or udnie) ")
    print("    --image: Path to image in JPEG format")
    print("    --output: Path to output image")
}

/// Startup parameters.
struct FastStyleTransferConfig {
    var style: String? = "candy"
    var image: String? = nil
    var output: String? = "out.jpg"
}

var config = FastStyleTransferConfig()
parseArguments(
    into: &config,
    with: [
        "style": \FastStyleTransferConfig.style,
        "image": \FastStyleTransferConfig.image,
        "output": \FastStyleTransferConfig.output,
    ]
)

guard let image = config.image, let output = config.output else {
    print("Error: No input image!")
    printUsage()
    exit(1)
}

guard FileManager.default.fileExists(atPath: image) else {
    print("Error: Failed to load image \(image). Check that the file exists and is in JPEG format.")
    printUsage()
    exit(1)
}

let imageTensor = Image(jpeg: URL(fileURLWithPath: image)).tensor / 255.0

// Init the model.
var style = TransformerNet()
do {
    try importWeights(&style, for: config.style!)
} catch {
    print("Error: Failed to load weights for style \(config.style!).")
    printUsage()
    exit(1)
}

// Apply the model to image.
let out = style(imageTensor.expandingShape(at: 0))

let outputImage = Image(tensor: out.squeezingShape(at: 0))
outputImage.save(to: URL(fileURLWithPath: output), colorspace: .rgb)

print("Writing output to \(output).")
