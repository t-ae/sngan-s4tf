import TensorFlow
import TensorBoardX

struct GBlock: Layer {
    enum UpSampleMethod: String, Codable {
        case nearestNeighbor, bilinear
    }
    
    @noDerivative
    let upsampleMethod: UpSampleMethod
    
    var conv1: SN<Conv2D<Float>>
    var conv2: SN<Conv2D<Float>>
    
    var convSC: SN<Conv2D<Float>>
    
    var norm1: XNorm
    var norm2: XNorm
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        upsampleMethod: UpSampleMethod,
        normalizationMethod: XNorm.Method
    ) {
        self.upsampleMethod = upsampleMethod
        
        let hiddenChannels = inputChannels
        
        conv1 = SN(Conv2D(filterShape: (3, 3, inputChannels, hiddenChannels), padding: .same,
                          filterInitializer: glorotUniform(scale: sqrt(2))))
        conv2 = SN(Conv2D(filterShape: (3, 3, hiddenChannels, outputChannels), padding: .same,
                          filterInitializer: glorotUniform(scale: sqrt(2))))
        
        norm1 = XNorm(method: normalizationMethod, dim: inputChannels)
        norm2 = XNorm(method: normalizationMethod, dim: hiddenChannels)
        
        convSC = SN(Conv2D(filterShape: (1, 1, inputChannels, outputChannels), padding: .same,
                           filterInitializer: glorotUniform(scale: 1)))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var res = input
        res = lrelu(norm1(res))
        res = conv1(upsample(res))
        res = lrelu(norm2(res))
        res = conv2(res)
        
        let shortcut = convSC(upsample(input))
        
        return res + shortcut
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
    
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        conv1.writeHistograms(writer: writer, tag: "\(tag).conv1", globalStep: globalStep)
        conv2.writeHistograms(writer: writer, tag: "\(tag).conv2", globalStep: globalStep)
        convSC.writeHistograms(writer: writer, tag: "\(tag).convSC", globalStep: globalStep)
    }
}

struct Generator: Layer {
    struct Options: Codable {
        var latentSize: Int
        var upsampleMethod: GBlock.UpSampleMethod
        var enableSpectralNorm: Bool
        var normalizationMethod: XNorm.Method
        var tanhOutput: Bool
    }
    
    @noDerivative
    var options: Options
    
    var head: SN<TransposedConv2D<Float>>
    var block1: GBlock
    var block2: GBlock
    var block3: GBlock
    var block4: GBlock
    var tail: SN<Conv2D<Float>>
    
    var bn: XNorm
    
    init(options: Options) {
        self.options = options
        
        head = SN(TransposedConv2D(filterShape: (4, 4, 128, options.latentSize),
                                   filterInitializer: glorotUniform(scale: 1)))
        block1 = GBlock(inputChannels: 128, outputChannels: 128,
                        upsampleMethod: options.upsampleMethod,
                        normalizationMethod: options.normalizationMethod)
        block2 = GBlock(inputChannels: 128, outputChannels: 64,
                        upsampleMethod: options.upsampleMethod,
                        normalizationMethod: options.normalizationMethod)
        block3 = GBlock(inputChannels: 64, outputChannels: 32,
                        upsampleMethod: options.upsampleMethod,
                        normalizationMethod: options.normalizationMethod)
        block4 = GBlock(inputChannels: 32, outputChannels: 16,
                        upsampleMethod: options.upsampleMethod,
                        normalizationMethod: options.normalizationMethod)
        tail = SN(Conv2D(filterShape: (3, 3, 16, 3), padding: .same,
                         filterInitializer: glorotUniform(scale: 1)))
        bn = XNorm(method: options.normalizationMethod, dim: 16)
    }
    
    mutating func preTrain() {
        if(options.enableSpectralNorm) {
            spectralNormalize(&self)
        }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = x.expandingShape(at: 1, 2) // [-1, 1, 1, latentSize]
        
        x = head(x) // [-1, 4, 4, 128]
        x = block1(x) // [-1, 8, 8, 128]
        x = block2(x) // [-1, 16, 16, 64]
        x = block3(x) // [-1, 32, 32, 32]
        x = block4(x) // [-1, 64, 64, 16]
        x = lrelu(bn(x))
        x = tail(x)
        
        if options.tanhOutput {
            x = tanh(x)
        }
        
        precondition(x.shape == [input.shape[0], 64, 64, 3], "Invalid shape: \(x.shape)")
        
        return x
    }
    
    func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        head.writeHistograms(writer: writer, tag: "G/head", globalStep: globalStep)
        block1.writeHistograms(writer: writer, tag: "G/block1", globalStep: globalStep)
        block2.writeHistograms(writer: writer, tag: "G/block2", globalStep: globalStep)
        block3.writeHistograms(writer: writer, tag: "G/block3", globalStep: globalStep)
        block4.writeHistograms(writer: writer, tag: "G/block4", globalStep: globalStep)
        tail.writeHistograms(writer: writer, tag: "G/tail", globalStep: globalStep)
    }
}
