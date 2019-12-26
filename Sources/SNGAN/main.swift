import Foundation
import TensorFlow
import TensorBoardX
import ImageLoader

Context.local.randomSeed = (42, 42)
let rng = XorshiftRandomNumberGenerator()

// MARK: - Configurations
let batchSize = 32
let latentSize = 128
let nDisUpdate = 5 // D/G training ratio

let generatorOptions = Generator.Options(
    latentSize: latentSize,
    upSampleMethod: .bilinear,
    enableSpectralNorm: true,
    normalizationMethod: .batchNorm,
    tanhOutput: false
)
let discriminatorOptions = Discriminator.Options(
    enableSpectralNorm: true,
    downSampleMethod: .avgPool,
    enableMinibatchStdConcat: true
)

//let lossObj = LSGANLoss()
//let lossObj = NonSaturatingLoss()
let lossObj = HingeLoss()

// MARK: - Model definition
var generator = Generator(options: generatorOptions)
var discriminator = Discriminator(options: discriminatorOptions)

let optG = Adam(for: generator, learningRate: 2e-4, beta1: 0.0, beta2: 0.9)
let optD = Adam(for: discriminator, learningRate: 2e-4, beta1: 0.0, beta2: 0.9)

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
    ],
    parallel: true,
    rng: rng
)
print("Total images: \(loader.entries.count)")

let seq = BatchImageSequence(loader: loader, batchSize: batchSize, infinite: true)

let plotGridCols = 8
let testNoise = sampleNoise(batchSize: plotGridCols*plotGridCols, latentSize: latentSize)

// MARK: - Plot
let logName = "\(lossObj.name)_\(generatorOptions.upSampleMethod.rawValue)_\(discriminatorOptions.downSampleMethod.rawValue)"
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
writer.addText(tag: "\(logName)/nDisUpdate", text: String(nDisUpdate))

// MARK: - Training loop
for (step, batch) in seq.enumerated() {
    if step % 10 == 0 {
        print("step: \(step)")
    }
    
    Context.local.learningPhase = .training
    
    var reals = batch.images
    reals = reals * 2 - 1
    
    // MARK: Train generator
    if step % nDisUpdate == 0 {
        let (lossG, 𝛁generator) = valueWithGradient(at: generator) { generator -> Tensor<Float> in
            let noises = sampleNoise(batchSize: batchSize, latentSize: latentSize)
            let fakes = generator(noises)
            let scores = discriminator(fakes)
            let loss = lossObj.lossG(scores)
            return loss
        }
        optG.update(&generator, along: 𝛁generator)
        writer.addScalar(tag: "lossG", scalar: lossG.scalarized(), globalStep: step)
    }
    
    // MARK: Train discrminator
    let noises = sampleNoise(batchSize: batchSize, latentSize: latentSize)
    let fakes = generator(noises)
    let (lossD, 𝛁discriminator) = valueWithGradient(at: discriminator) { discriminator -> Tensor<Float> in
        let fakeScores = discriminator(fakes)
        
        let realScores = discriminator(reals)
        let loss = lossObj.lossD(real: realScores, fake: fakeScores)
        return loss
    }
    optD.update(&discriminator, along: 𝛁discriminator)
    writer.addScalar(tag: "lossD", scalar: lossD.scalarized(), globalStep: step)
    
    if step % 500 == 0 {
        plotImages(tag: "reals", images: reals, globalStep: step)
        plotImages(tag: "fakes", images: fakes, globalStep: step)
        
        generator.writeHistograms(writer: writer, globalStep: step)
        discriminator.writeHistograms(writer: writer, globalStep: step)
        
        let fakeStd = fakes.standardDeviation(alongAxes: 0).mean().scalarized()
        writer.addScalar(tag: "fake_std", scalar: fakeStd, globalStep: step)
        
        let grad = gradient(at: fakes) { fakes -> Tensor<Float> in
            let scores = discriminator(fakes)
            return scores.sum()
        }
        let gradnorm = sqrt(grad.squared().sum())
        writer.addScalar(tag: "gradnorm", scalar: gradnorm.scalarized(), globalStep: step)
        
        writer.flush()
    }
    
    if step % 5000 == 0 {
        // Inference
        Context.local.learningPhase = .inference
        let testImage = generator(testNoise)
        plotImages(tag: "tests", images: testImage, globalStep: step)
        writer.flush()
    }
}

writer.close()
