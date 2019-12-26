import TensorFlow
import TensorBoardX
import CustomLayers

struct DBlock: Layer {
    enum DownSampleMethod: String, Codable {
        case avgPool
    }
    
    @noDerivative
    let downSampleMethod: DownSampleMethod
    
    var conv1: SNConv2D<Float>
    var conv2: SNConv2D<Float>
    
    var convSC: SNConv2D<Float>
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        enableSpectralNorm: Bool,
        downSampleMethod: DownSampleMethod
    ) {
        self.downSampleMethod = downSampleMethod
        
        let hiddenChannels = inputChannels
        
        conv1 = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, hiddenChannels), padding: .same,
                                filterInitializer: glorotUniform(scale: sqrt(2))),
                         enabled: enableSpectralNorm)
        conv2 = SNConv2D(Conv2D(filterShape: (3, 3, hiddenChannels, outputChannels), padding: .same,
                                filterInitializer: glorotUniform(scale: sqrt(2))),
                         enabled: enableSpectralNorm)
        
        convSC = SNConv2D(Conv2D(filterShape: (1, 1, inputChannels, outputChannels), padding: .same,
                                 filterInitializer: glorotUniform()),
                          enabled: enableSpectralNorm)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var res = input
        res = conv1(lrelu(res))
        res = conv2(lrelu(res))
        res = downsample(res)
        
        let shortcut = convSC(downsample(input))
        
        return res + shortcut
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
        var enableMinibatchStdConcat: Bool
    }
    
    @noDerivative
    var options: Options
    
    var head: SNConv2D<Float>
    var block1: DBlock
    var block2: DBlock
    var block3: DBlock
    var block4: DBlock
    var tail: SNConv2D<Float>
    
    var stdConcat: MinibatchStdConcat<Float>
    
    init(options: Options) {
        self.options = options
        
        head = SNConv2D(Conv2D(filterShape: (1, 1, 3, 16), filterInitializer: glorotUniform()),
                        enabled: options.enableSpectralNorm)
        
        block1 = DBlock(inputChannels: 16, outputChannels: 32,
                        enableSpectralNorm: options.enableSpectralNorm,
                        downSampleMethod: options.downSampleMethod)
        block2 = DBlock(inputChannels: 32, outputChannels: 64,
                        enableSpectralNorm: options.enableSpectralNorm,
                        downSampleMethod: options.downSampleMethod)
        block3 = DBlock(inputChannels: 64, outputChannels: 128,
                        enableSpectralNorm: options.enableSpectralNorm,
                        downSampleMethod: options.downSampleMethod)
        block4 = DBlock(inputChannels: 128, outputChannels: 128,
                        enableSpectralNorm: options.enableSpectralNorm,
                        downSampleMethod: options.downSampleMethod)
        
        let stdDim = options.enableMinibatchStdConcat ? 1 : 0
        stdConcat = MinibatchStdConcat(groupSize: 4)
        tail = SNConv2D(Conv2D(filterShape: (4, 4, 128 + stdDim, 1), filterInitializer: glorotUniform()),
                        enabled: options.enableSpectralNorm)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = lrelu(head(x)) // [-1, 64, 64, 16]
        x = block1(x) // [-1, 32, 32, 32]
        x = block2(x) // [-1, 16, 16, 64]
        x = block3(x) // [-1, 8, 8, 128]
        x = block4(x) // [-1, 4, 4, 128]
        x = lrelu(x)
        
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
        writer.addHistograms(tag: "D/block4", layer: block4, globalStep: globalStep)
        writer.addHistograms(tag: "D/tail", layer: tail, globalStep: globalStep)
    }
}
