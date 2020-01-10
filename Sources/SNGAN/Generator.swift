import TensorFlow
import TensorBoardX
import CustomLayers

struct GBlock: Layer {
    @noDerivative
    let upsampleMethod: Resize.Method
    
    var conv: SNConv2D<Float>
    var norm: XNorm
    
    var activation: Activation
    
    var resize: Resize
    
    init(
        inputShape: (Int, Int, Int),
        outputChannels: Int,
        enableSpectralNorm: Bool,
        upsampleMethod: Resize.Method,
        normalizationMethod: XNorm.Method,
        activation: Activation,
        avoidPadding: Bool
    ) {
        self.upsampleMethod = upsampleMethod
        
        let pad: Padding = avoidPadding ? .valid : .same
        
        conv = SNConv2D(Conv2D(filterShape: (3, 3, inputShape.2, outputChannels), padding: pad,
                                filterInitializer: heNormal()),
                         enabled: enableSpectralNorm)
        
        norm = XNorm(method: normalizationMethod, dim: outputChannels)
        
        self.activation = activation
        
        // +2 to avoid padding
        let resizeW = inputShape.1 * 2 + (avoidPadding ? 2 : 0)
        let resizeH = inputShape.0 * 2 + (avoidPadding ? 2 : 0)
        self.resize = Resize(width: resizeW, height: resizeH, method: upsampleMethod, alignCorners: true)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = resize(x)
        x = conv(x)
        x = norm(x)
        x = activation(x)
        
        return x
    }
}

struct Generator: Layer {
    struct Options: Codable {
        var latentSize: Int
        var upSampleMethod: Resize.Method
        var enableSpectralNorm: Bool
        var normalizationMethod: XNorm.Method
        var activation: Activation.Method
        var tanhOutput: Bool
        var avoidPadding: Bool
    }
    
    @noDerivative
    var options: Options
    
    var head: SNDense<Float>
    var block1: GBlock
    var block2: GBlock
    var block3: GBlock
    var block4: GBlock
    var tail: SNConv2D<Float>
    
    var activation: Activation
    
    init(options: Options) {
        self.options = options
        self.activation = Activation(method: options.activation)
        
        head = SNDense(Dense(inputSize: options.latentSize, outputSize: 4*4*128,
                             weightInitializer: heNormal()),
                       enabled: options.enableSpectralNorm)
        block1 = GBlock(inputShape: (4, 4, 128), outputChannels: 128,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod,
                        activation: activation,
                        avoidPadding: options.avoidPadding)
        block2 = GBlock(inputShape: (8, 8, 128), outputChannels: 64,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod,
                        activation: activation,
                        avoidPadding: options.avoidPadding)
        block3 = GBlock(inputShape: (16, 16, 64), outputChannels: 32,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod,
                        activation: activation,
                        avoidPadding: options.avoidPadding)
        block4 = GBlock(inputShape: (32, 32, 32), outputChannels: 16,
                        enableSpectralNorm: options.enableSpectralNorm,
                        upsampleMethod: options.upSampleMethod,
                        normalizationMethod: options.normalizationMethod,
                        activation: activation,
                        avoidPadding: options.avoidPadding)
        
        // SN disabled
        tail = SNConv2D(Conv2D(filterShape: (3, 3, 16, 3), padding: .same,
                               filterInitializer: heNormal()),
                        enabled: false)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = activation(head(x)) // [-1, 4*4*128]
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
