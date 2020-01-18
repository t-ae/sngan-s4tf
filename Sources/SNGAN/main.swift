import Foundation
import TensorFlow
import TensorBoardX
import ImageLoader

Context.local.randomSeed = (42, 42)
let rng = XorshiftRandomNumberGenerator()

// MARK: - Configurations
let config = Config(
    batchSize: 64,
    nDisUpdate: 3,
    loss: .hinge,
    imageSize: .x64
)

let genOptions = Generator.Options(
    latentSize: 128,
    upSampleMethod: .bilinear,
    enableSpectralNorm: true,
    normalizationMethod: .batchNorm,
    activation: .elu,
    tanhOutput: false,
    avoidPadding: false
)
let discOptions = Discriminator.Options(
    enableSpectralNorm: true,
    downSampleMethod: .avgPool,
    normalizationMethod: .none,
    activation: .elu,
    enableMinibatchStdConcat: true
)

let criterion = GANLoss(type: config.loss)

// MARK: - Model definition
var generator = Generator(imageSize: config.imageSize, options: genOptions)
var discriminator = Discriminator(imageSize: config.imageSize, options: discOptions)

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
        Transforms.resizeBilinear(aspectFill: config.imageSize.rawValue),
        Transforms.centerCrop(width: config.imageSize.rawValue, height: config.imageSize.rawValue),
//        Transforms.paddingToSquare(with: 1),
//        Transforms.resizeBilinear(aspectFill: config.imageSize.rawValue),
//        Transforms.randomFlipHorizontally(),
    ],
    rng: rng
)
print("Total images: \(loader.entries.count)")


let testNoise = sampleNoise(batchSize: 64, latentSize: genOptions.latentSize)
let testNoiseIntpl = sampleInterpolationNoise(latentSize: genOptions.latentSize)

// MARK: - Plot
let logName = "\(config.loss.rawValue)_\(genOptions.upSampleMethod.rawValue)_\(discOptions.downSampleMethod.rawValue)_\(config.imageSize.rawValue)"
let writer = SummaryWriter(logdir: URL(fileURLWithPath: "./output/\(logName)"))
func plotImages(tag: String, images: Tensor<Float>,
                colSize: Int = config.imageSize.plotGridCols,  globalStep: Int) {
    var images = images
    images = (images + 1) / 2
    images = images.clipped(min: 0, max: 1)
    writer.addImages(tag: tag,
                     images: images,
                     colSize: colSize,
                     globalStep: globalStep)
}

// Write configurations
try writer.addJSONText(tag: "\(logName)/config", encodable: config)
try writer.addJSONText(tag: "\(logName)/generatorOptions", encodable: genOptions)
try writer.addJSONText(tag: "\(logName)/discriminatorOptions", encodable: discOptions)

// MARK: - Training loop
var step = 0
for epoch in 0..<1000000 {
    print("Epoch: \(epoch)")
    loader.shuffle()
    
    for batch in loader.iterator(batchSize: config.batchSize) {
        defer { step += 1 }
        if step % 10 == 0 {
            print("step: \(step)")
        }
        
        Context.local.learningPhase = .training
        
        var reals = batch.images
        reals = reals * 2 - 1
        
        // MARK: Train generator
        if step % config.nDisUpdate == 0 {
            let (lossG, 𝛁generator) = valueWithGradient(at: generator) { generator -> Tensor<Float> in
                let noises = sampleNoise(batchSize: config.batchSize, latentSize: genOptions.latentSize)
                let fakes = generator(noises)
                let scores = discriminator(fakes)
                let loss = criterion.lossG(scores)
                return loss
            }
            optG.update(&generator, along: 𝛁generator)
            writer.addScalar(tag: "Loss/G", scalar: lossG.scalarized(), globalStep: step)
        }
        
        // MARK: Train discrminator
        let noises = sampleNoise(batchSize: config.batchSize, latentSize: genOptions.latentSize)
        let fakes = generator(noises)
        let (lossD, 𝛁discriminator) = valueWithGradient(at: discriminator) { discriminator -> Tensor<Float> in
            let fakeScores = discriminator(fakes)
            
            let realScores = discriminator(reals)
            let loss = criterion.lossD(real: realScores, fake: fakeScores)
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
            plotImages(tag: "test/intpl", images: intplImage, colSize: 8, globalStep: step)
            writer.flush()
        }
    }
}


writer.close()
