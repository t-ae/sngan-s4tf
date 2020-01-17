import Foundation
import TensorFlow
import TensorBoardX
import ImageLoader

Context.local.randomSeed = (42, 42)
let rng = XorshiftRandomNumberGenerator()

// MARK: - Configurations
let batchSize = 64
let latentSize = 128
let nDisUpdate = 3 // D/G training ratio

let generatorOptions = Generator.Options(
    latentSize: latentSize,
    upSampleMethod: .bilinear,
    enableSpectralNorm: true,
    normalizationMethod: .batchNorm,
    activation: .elu,
    tanhOutput: false,
    avoidPadding: false
)
let discriminatorOptions = Discriminator.Options(
    enableSpectralNorm: true,
    downSampleMethod: .avgPool,
    normalizationMethod: .none,
    activation: .elu,
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
let entries = [Entry](directory: imageDir)
let loader = ImageLoader(
    entries: entries,
    transforms: [
        Transforms.resizeBilinear(aspectFill: 64),
        Transforms.centerCrop(width: 64, height: 64)
    ],
    rng: rng
)
print("Total images: \(loader.entries.count)")


let plotGridCols = 8
let testNoise = sampleNoise(batchSize: plotGridCols*plotGridCols, latentSize: latentSize)
let testNoiseIntpl = sampleInterpolationNoise(latentSize: latentSize)

// MARK: - Plot
let logName = "\(lossObj.name)_\(generatorOptions.upSampleMethod.rawValue)_\(discriminatorOptions.downSampleMethod.rawValue)"
let writer = SummaryWriter(logdir: URL(fileURLWithPath: "./output/\(logName)"))
func plotImages(tag: String, images: Tensor<Float>, globalStep: Int) {
    var images = images
    images = (images + 1) / 2
    images = images.clipped(min: 0, max: 1)
    writer.addImages(tag: tag, images: images, colSize: plotGridCols, globalStep: globalStep)
}

// Write configurations
writer.addText(tag: "\(logName)/loss", text: lossObj.name)
writer.addText(tag: "\(logName)/generatorOptions", text: generatorOptions.prettyJsonString())
writer.addText(tag: "\(logName)/discriminatorOptions", text: discriminatorOptions.prettyJsonString())
writer.addText(tag: "\(logName)/nDisUpdate", text: String(nDisUpdate))

// MARK: - Training loop
var step = 0
for epoch in 0..<1000000 {
    print("Epoch: \(epoch)")
    loader.shuffle()
    
    for batch in loader.iterator(batchSize: batchSize) {
        defer { step += 1 }
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
            writer.addScalar(tag: "Loss/G", scalar: lossG.scalarized(), globalStep: step)
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
        writer.addScalar(tag: "Loss/D", scalar: lossD.scalarized(), globalStep: step)
        
        if step % 500 == 0 {
            plotImages(tag: "reals", images: reals, globalStep: step)
            plotImages(tag: "fakes", images: fakes, globalStep: step)
            
            generator.writeHistograms(writer: writer, globalStep: step)
            discriminator.writeHistograms(writer: writer, globalStep: step)
            
            let fakeStd = fakes.standardDeviation(alongAxes: 0).mean().scalarized()
            writer.addScalar(tag: "Value/fake_std", scalar: fakeStd, globalStep: step)
            
            let grad = gradient(at: fakes) { fakes -> Tensor<Float> in
                let scores = discriminator(fakes)
                return scores.sum()
            }
            let gradnorm = sqrt(grad.squared().sum(squeezingAxes: [1, 2, 3])).mean()
            writer.addScalar(tag: "Value/gradnorm", scalar: gradnorm.scalarized(), globalStep: step)
            
            writer.flush()
        }
        
        if step % 5000 == 0 {
            // Inference
            Context.local.learningPhase = .inference
            let testImage = generator(testNoise)
            plotImages(tag: "test/random", images: testImage, globalStep: step)
            
            let intplImage = generator(testNoiseIntpl)
            plotImages(tag: "test/intpl", images: intplImage, globalStep: step)
            writer.flush()
        }
    }
}


writer.close()
