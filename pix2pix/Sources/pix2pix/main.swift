import TensorFlow
import Files
import Foundation
import TensorBoardX
import ArgumentParser

struct Options: ParsableArguments {
    @Option(default: "/notebooks/avolodin/data/facades/", help: ArgumentHelp("Path to the splitted dataset folder", valueName: "dataset"))
    var datasetPath: String
    
    @Option(default: 3, help: ArgumentHelp("GPU Index", valueName: "gpu-index"))
    var gpuIndex: UInt
    
    @Option(default: 3, help: ArgumentHelp("Number of epochs", valueName: "epochs"))
    var epochs: Int
    
    @Option(default: "/tmp/tensorboardx", help: ArgumentHelp("TensorBoard logdir path", valueName: "tensorboard-logdir"))
    var tensorboardLogdir: String
}

let options = Options.parseOrExit()

let logdir = URL(fileURLWithPath: options.tensorboardLogdir).appendingPathComponent(String(Int(Date().timeIntervalSince1970)))
//try? FileManager.default.removeItem(at: logdir)
let writer = SummaryWriter(logdir: logdir)

let facadesFolder = try Folder(path: options.datasetPath)
let trainFolder = try facadesFolder.subfolder(named: "train")
let testFolder = try facadesFolder.subfolder(named: "test")
let trainDataset = try Facades(folder: trainFolder)
let testDataset = try Facades(folder: testFolder)

var generator = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, useDropout: false)
var discriminator = NetD(inChannels: 6, lastConvFilters: 64)

let optimizerG = Adam(for: generator, learningRate: 0.0002, beta1: 0.5)
let optimizerD = Adam(for: discriminator, learningRate: 0.0002, beta1: 0.5)

let epochs = 10000
let batchSize = 1
let lambdaL1 = Tensorf(100)
let zeros = Tensorf(0)
let ones = Tensorf(1)
let gpuIndex = options.gpuIndex

for epoch in 0..<epochs {
    print("Epoch \(epoch) started at: \(Date())")
    Context.local.learningPhase = .training
    
    let trainingShuffled = trainDataset.dataset.shuffled(sampleCount: trainDataset.count, randomSeed: Int64(epoch))
    
    var discriminatorTotalLoss = Tensorf(0)
    var generatorTotalLoss = Tensorf(0)
    var discriminatorCount = 0
    
    for batch in trainingShuffled.batched(batchSize) {
        // we do it outside of GPU scope so that dataset shuffling happens on CPU side
        let concatanatedImages = batch.sourceImages.concatenated(with: batch.targetImages)
        
        withDevice(.gpu, gpuIndex) {
            let scaledImages = _Raw.resizeNearestNeighbor(images: concatanatedImages, size: [286, 286])
            var croppedImages = scaledImages.slice(lowerBounds: Tensor<Int32>([0, Int32(random() % 30), Int32(random() % 30), 0]),
                                                   sizes: [2, 256, 256, 3])
            if random() % 2 == 0 {
                croppedImages = _Raw.reverse(croppedImages, dims: [false, false, true, false])
            }
        
            let sourceImages = croppedImages[0].expandingShape(at: 0)
            let targetImages = croppedImages[1].expandingShape(at: 0)
            
            let generatorGradient = TensorFlow.gradient(at: generator) { g -> Tensorf in
                let fakeImages = g(sourceImages)
                let fakeAB = sourceImages.concatenated(with: fakeImages, alongAxis: 3)
                let fakePrediction = discriminator(fakeAB)
        
                let ganLoss = sigmoidCrossEntropy(logits: fakePrediction, 
                                                  labels: ones.broadcasted(to: fakePrediction.shape))
                let l1Loss = meanAbsoluteError(predicted: fakeImages, 
                                               expected: targetImages) * lambdaL1
        
                generatorTotalLoss += ganLoss + l1Loss
                return ganLoss + l1Loss
            }
           
            let fakeImages = generator(sourceImages)
            let descriminatorGradient = TensorFlow.gradient(at: discriminator) { d -> Tensorf in
                let fakeAB = sourceImages.concatenated(with: fakeImages, 
                                                       alongAxis: 3)
                let fakePrediction = d(fakeAB)
                let fakeLoss = sigmoidCrossEntropy(logits: fakePrediction, 
                                                   labels: zeros.broadcasted(to: fakePrediction.shape))
    
                let realAB = sourceImages.concatenated(with: targetImages, 
                                                       alongAxis: 3)
                let realPrediction = d(realAB)
                let realLoss = sigmoidCrossEntropy(logits: realPrediction, 
                                                   labels: ones.broadcasted(to: fakePrediction.shape))
        
                discriminatorTotalLoss += (fakeLoss + realLoss) * 0.5
                                                                                
                return (fakeLoss + realLoss) * 0.5
            }
            
            optimizerG.update(&generator, along: generatorGradient)
            optimizerD.update(&discriminator, along: descriminatorGradient)
        }
        
        discriminatorCount += 1
    }
    let generatorLoss = generatorTotalLoss / Float(discriminatorCount)
    let discriminatorLoss = discriminatorTotalLoss / Float(discriminatorCount)
    writer.addScalars(mainTag: "train_loss", 
                      taggedScalars: ["Generator": generatorLoss.scalars[0], 
                                      "Discriminator": discriminatorLoss.scalars[0]], 
                      globalStep: epoch)
    
    Context.local.learningPhase = .inference
    
    var totalLoss = Tensorf(0)
    var count = 0
    
    for batch in testDataset.dataset.batched(1) {
        let fakeImages = generator(batch.sourceImages)


        let tensorImage = batch.sourceImages
                               .concatenated(with: fakeImages,
                                             alongAxis: 2) / 2.0 + 0.5
        // for some reason it's just doesn't work for now
        /*writer.addImages(tag: "images",
                           images: tensorImage,
                           globalStep: epoch)*/
        let subfolder = try Folder.current.createSubfolder(named: "inference_\(epoch)")
        let image = Image(tensor: (tensorImage * 255)[0])
        let saveURL = subfolder.url.appendingPathComponent("\(count).jpg", isDirectory: false)
        image.save(to: saveURL, format: .rgb)
                        
        let ganLoss = sigmoidCrossEntropy(logits: fakeImages, 
                                          labels: ones.broadcasted(to: fakeImages.shape))
        let l1Loss = meanAbsoluteError(predicted: fakeImages, 
                                       expected: batch.targetImages) * lambdaL1
        
        totalLoss += ganLoss + l1Loss
        count += 1
        
        if count == 5 {
            break
        }
    }
    
    let testLoss = totalLoss / Float(count)
    writer.addScalars(mainTag: "test_loss", 
                      taggedScalars: ["Generator": testLoss.scalars[0]],
                      globalStep: epoch)
}
