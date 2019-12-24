import Foundation
import TensorFlow
import TensorBoardX
import ImageLoader

// MARK: - Configurations
let batchSize = 32
let latentSize = 128

let generatorOptions = Generator.Options(
    latentSize: latentSize,
    upsampleMethod: .bilinear,
    enableSpectralNorm: true,
    normMethod: .batchNorm
)
let discriminatorOptions = Discriminator.Options(
    downsampleMethod: .avgPool,
    enableSpectralNorm: true,
    normMethod: .batchNorm,
    enableMinibatchStdConcat: true
)

//let lossObj = LSGANLoss()
//let lossObj = NonSaturatingLoss()
let lossObj = HingeLoss()

// MARK: - Model definition
var generator = Generator(options: generatorOptions)
var discriminator = Discriminator(options: discriminatorOptions)

let optG = Adam(for: generator, learningRate: 2e-4, beta1: 0.5)
let optD = Adam(for: discriminator, learningRate: 2e-4, beta1: 0.5)

// MARK: - Train/test data
let args = ProcessInfo.processInfo.arguments
guard args.count == 2 else {
    print("Image directory is not specified.")
    exit(1)
}
print("Seaerch images...")
let imageDir = URL(fileURLWithPath: args[1])
let loader = try ImageLoader(
    directory: imageDir,
    transforms: [
        Transforms.resizeBilinear(aspectFill: 64),
        Transforms.centerCrop(width: 64, height: 64)
    ]
)
print("Total images: \(loader.entries.count)")

let plotGridCols = 8
let testNoise = sampleNoise(batchSize: plotGridCols*plotGridCols, latentSize: latentSize)

// MARK: - Plot
let logName = "\(lossObj.name)_\(generatorOptions.upsampleMethod.rawValue)_\(discriminatorOptions.downsampleMethod.rawValue)"
let writer = SummaryWriter(logdir: URL(fileURLWithPath: "./output/\(logName)"))
func plotImages(tag: String, images: Tensor<Float>, globalStep: Int) {
    let height = images.shape[1]
    let width = images.shape[2]
    let plotGridRows = images.shape[0] / plotGridCols
    var grid = images.reshaped(to: [plotGridRows, plotGridCols, height, width, 3])
    grid = grid.transposed(permutation: [0, 2, 1, 3, 4])
    grid = grid.reshaped(to: [height*plotGridRows, width*plotGridCols, 3])
    grid = (grid + 1) / 2
    grid = grid.clipped(min: 0, max: 1)
    writer.addImage(tag: tag, image: grid, globalStep: globalStep)
}

// Write configurations
writer.addText(tag: "\(logName)/loss", text: lossObj.name)
writer.addText(tag: "\(logName)/generatorOptions", text: generatorOptions.prettyJsonString())
writer.addText(tag: "\(logName)/discriminatorOptions", text: discriminatorOptions.prettyJsonString())

// MARK: - Training loop
for step in 0..<10_000_000 {
    Context.local.learningPhase = .training
    
    var (reals, _) = loader.nextBatch(size: batchSize)
    reals = reals * 2 - 1
    
    let noises = sampleNoise(batchSize: batchSize, latentSize: latentSize)
    
    // MARK: Train generator
    generator.preTrain()
    let (lossG, 𝛁generator) = valueWithGradient(at: generator) { generator -> Tensor<Float> in
        let fakes = generator(noises)
        let scores = discriminator(fakes)
        let loss = lossObj.lossG(scores)
        return loss
    }
    optG.update(&generator, along: 𝛁generator)
    
    // MARK: Train discrminator
    let fakes = generator(noises)
    discriminator.preTrain()
    let (lossD, 𝛁discriminator) = valueWithGradient(at: discriminator) { discriminator -> Tensor<Float> in
        let fakeScores = discriminator(fakes)
        
        let realScores = discriminator(reals)
        let loss = lossObj.lossD(real: realScores, fake: fakeScores)
        return loss
    }
    optD.update(&discriminator, along: 𝛁discriminator)
    
    if step % 10 == 0 {
        print("\(step): lossG=\(lossG.scalarized()), lossD=\(lossD.scalarized())")
    }
    writer.addScalar(tag: "lossG", scalar: lossG.scalarized(), globalStep: step)
    writer.addScalar(tag: "lossD", scalar: lossD.scalarized(), globalStep: step)
    
    if step % 100 == 0 {
        plotImages(tag: "reals", images: reals, globalStep: step)
        plotImages(tag: "fakes", images: fakes, globalStep: step)
        
        generator.writeHistograms(writer: writer, globalStep: step)
        discriminator.writeHistograms(writer: writer, globalStep: step)
        
        let fakeStd = fakes.standardDeviation(alongAxes: 0).mean().scalarized()
        writer.addScalar(tag: "fake_std", scalar: fakeStd, globalStep: step)
        
        writer.flush()
    }
    
    if step % 1000 == 0 {
        // Inference
        Context.local.learningPhase = .inference
        let testImage = generator(testNoise)
        plotImages(tag: "tests", images: testImage, globalStep: step)
        writer.flush()
    }
}
