import TensorFlow
import TensorBoardX
import CustomLayers

struct GBlock: Layer {
    enum UpSampleMethod: String, Codable {
        case nearestNeighbor, bilinear
    }
    
    @noDerivative
    let upsampleMethod: UpSampleMethod
    
    var conv: SNConv2D<Float>
    var norm: XNorm
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        enableSpectralNorm: Bool,
        upsampleMethod: UpSampleMethod,
        normalizationMethod: XNorm.Method
    ) {
        self.upsampleMethod = upsampleMethod
        
        conv = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, outputChannels), padding: .same,
                                filterInitializer: glorotUniform(scale: sqrt(2))),
                         enabled: enableSpectralNorm)
        
        norm = XNorm(method: normalizationMethod, dim: outputChannels)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = upsample(x)
        x = conv(x)
        x = norm(x)
        x = lrelu(x)
        
        return x
    }
    
    var upsampling = UpSampling2D<Float>(size: 2)
    
    @differentiable
    func upsample(_ x: Tensor<Float>) -> Tensor<Float> {
        switch upsampleMethod {
        case .nearestNeighbor:
            return upsampling(x)
        case .bilinear:
            return resize2xBilinear(images: x)
        }
    }
}

struct Generator: Layer {
    struct Options: Codable {
        var latentSize: Int
        var upSampleMethod: GBlock.UpSampleMethod
        var enableSpectralNorm: Bool
        var normalizationMethod: XNorm.Method
        var tanhOutput: Bool
    }
    
    @noDerivative
    var options: Options
    
    var head: SNDense<Float>
    var block1: GBlock
    var block2: GBlock
    var block3: GBlock
    var block4: GBlock
    var tail: SNConv2D<Float>
    
    init(options: Options) {
        self.options = options
        
        head = SNDense(Dense(inputSize: options.latentSize, outputSize: 4*4*128,
                             weightInitializer: glorotUniform()),
                       enabled: options.enableSpectralNorm)
        block1 = GBlock(inputChannels: 128, outputChannels: 128,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod)
        block2 = GBlock(inputChannels: 128, outputChannels: 64,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod)
        block3 = GBlock(inputChannels: 64, outputChannels: 32,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod)
        block4 = GBlock(inputChannels: 32, outputChannels: 16,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod)
        
        // SN disabled
        tail = SNConv2D(Conv2D(filterShape: (3, 3, 16, 3), padding: .same,
                               filterInitializer: glorotUniform()),
                        enabled: false)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = lrelu(head(x)) // [-1, 4*4*128]
        x = x.reshaped(to: [-1, 4 ,4, 128])
        x = block1(x) // [-1, 8, 8, 128]
        x = block2(x) // [-1, 16, 16, 64]
        x = block3(x) // [-1, 32, 32, 32]
        x = block4(x) // [-1, 64, 64, 16]
        x = tail(x)
        
        if options.tanhOutput {
            x = tanh(x)
        }
        
        precondition(x.shape == [input.shape[0], 64, 64, 3], "Invalid shape: \(x.shape)")
        
        return x
    }
    
    func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        writer.addHistograms(tag: "G/head", layer: head, globalStep: globalStep)
        writer.addHistograms(tag: "G/block1", layer: block1, globalStep: globalStep)
        writer.addHistograms(tag: "G/block2", layer: block2, globalStep: globalStep)
        writer.addHistograms(tag: "G/block3", layer: block3, globalStep: globalStep)
        writer.addHistograms(tag: "G/block4", layer: block4, globalStep: globalStep)
        writer.addHistograms(tag: "G/tail", layer: tail, globalStep: globalStep)
    }
}
