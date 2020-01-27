import TensorFlow
import TensorBoardX
import GANUtils

struct DBlock: Layer {
    enum DownSampleMethod: String, Codable {
        case avgPool
    }
    
    @noDerivative
    let downSampleMethod: DownSampleMethod
    
    var conv1: SNConv2D<Float>
    var conv2: SNConv2D<Float>
    var norm1: XNorm
    var norm2: XNorm
    
    var activation: Activation
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        enableSpectralNorm: Bool,
        normalizationMethod: XNorm.Method,
        downSampleMethod: DownSampleMethod,
        activation: Activation
    ) {
        self.downSampleMethod = downSampleMethod
        
        conv1 = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, outputChannels), padding: .same,
                                filterInitializer: heNormal()),
                         enabled: enableSpectralNorm)
        conv2 = SNConv2D(Conv2D(filterShape: (3, 3, outputChannels, outputChannels), padding: .same,
                                filterInitializer: heNormal()),
                         enabled: enableSpectralNorm)
        
        norm1 = XNorm(method: normalizationMethod, dim: outputChannels)
        norm2 = XNorm(method: normalizationMethod, dim: outputChannels)
        
        self.activation = activation
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(x)
        x = norm1(x)
        x = activation(x)
        x = conv2(x)
        x = norm2(x)
        x = activation(x)
        x = downsample(x)
        return x
    }
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @differentiable
    func downsample(_ x: Tensor<Float>) -> Tensor<Float> {
        switch downSampleMethod {
        case .avgPool:
            return avgPool(x)
        }
    }
}

struct Discriminator: Layer {
    struct Options: Codable {
        var enableSpectralNorm: Bool
        var downSampleMethod: DBlock.DownSampleMethod
        var normalizationMethod: XNorm.Method
        var activation: Activation.Method
        var enableMinibatchStdConcat: Bool
    }
    
    @noDerivative
    var imageSize: ImageSize
    
    @noDerivative
    var options: Options
    
    var head: SNConv2D<Float>
    var block1: DBlock
    var block2: DBlock
    var block3: DBlock
    var block4: DBlock
    var block5: DBlock
    var tail: SNConv2D<Float>
    
    var stdConcat: MinibatchStdConcat<Float>
    
    var activation: Activation
    
    init(imageSize: ImageSize, options: Options) {
        self.imageSize = imageSize
        self.options = options
        self.activation = Activation(method: options.activation)
        
        let firstChannels = 32 * 32 / imageSize.rawValue
        
        head = SNConv2D(Conv2D(filterShape: (1, 1, 3, firstChannels), filterInitializer: heNormal()),
                        enabled: options.enableSpectralNorm)
        
        block1 = DBlock(inputChannels: firstChannels, outputChannels: firstChannels*2,
                        enableSpectralNorm: options.enableSpectralNorm,
                        normalizationMethod: options.normalizationMethod,
                        downSampleMethod: options.downSampleMethod,
                        activation: activation)
        block2 = DBlock(inputChannels: firstChannels*2, outputChannels: firstChannels*4,
                        enableSpectralNorm: options.enableSpectralNorm,
                        normalizationMethod: options.normalizationMethod,
                        downSampleMethod: options.downSampleMethod,
                        activation: activation)
        block3 = DBlock(inputChannels: firstChannels*4, outputChannels: firstChannels*8,
                        enableSpectralNorm: options.enableSpectralNorm,
                        normalizationMethod: options.normalizationMethod,
                        downSampleMethod: options.downSampleMethod,
                        activation: activation)
        block4 = DBlock(inputChannels: firstChannels*8, outputChannels: firstChannels*16,
                        enableSpectralNorm: options.enableSpectralNorm,
                        normalizationMethod: options.normalizationMethod,
                        downSampleMethod: options.downSampleMethod,
                        activation: activation)
        block5 = DBlock(inputChannels: firstChannels*16, outputChannels: firstChannels*32,
                        enableSpectralNorm: options.enableSpectralNorm,
                        normalizationMethod: options.normalizationMethod,
                        downSampleMethod: options.downSampleMethod,
                        activation: activation)
        
        // Output channels will be 256 whatever `imageSize` is.
        
        let stdDim = options.enableMinibatchStdConcat ? 1 : 0
        stdConcat = MinibatchStdConcat(groupSize: 4)
        tail = SNConv2D(Conv2D(filterShape: (4, 4, 256 + stdDim, 1),
                               filterInitializer: heNormal()),
                        enabled: options.enableSpectralNorm)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = activation(head(x))
        x = block1(x)
        x = block2(x)
        x = block3(x)
        if imageSize >= .x64 {
            x = block4(x)
        }
        if imageSize >= .x128 {
            x = block5(x) 
        }
        
        if options.enableMinibatchStdConcat {
            x = stdConcat(x)  // [-1, 4, 4, 129]
        }
        
        x = tail(x) // [-1, 1, 1, 1]
        x = x.squeezingShape(at: [1, 2])
        
        return x
    }

    func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        writer.addHistograms(tag: "D/head", layer: head, globalStep: globalStep)
        writer.addHistograms(tag: "D/block1", layer: block1, globalStep: globalStep)
        writer.addHistograms(tag: "D/block2", layer: block2, globalStep: globalStep)
        writer.addHistograms(tag: "D/block3", layer: block3, globalStep: globalStep)
        if imageSize >= .x64 {
            writer.addHistograms(tag: "D/block4", layer: block4, globalStep: globalStep)
        }
        if imageSize >= .x128 {
            writer.addHistograms(tag: "D/block4", layer: block4, globalStep: globalStep)
        }
        writer.addHistograms(tag: "D/tail", layer: tail, globalStep: globalStep)
    }
}
