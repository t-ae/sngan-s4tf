import TensorFlow
import TensorBoardX

struct Generator: Layer {
    struct Options: Codable {
        var latentSize: Int
        var upsampleMethod: UpSamplingConv2D.Method
        var enableSpectralNorm: Bool
        var normMethod: XNorm.Method
        var tanhOutput: Bool
    }
    
    @noDerivative
    var options: Options
    
    var head: SN<TransposedConv2D<Float>>
    var conv1: UpSamplingConv2D
    var conv2: UpSamplingConv2D
    var conv3: UpSamplingConv2D
    var conv4: UpSamplingConv2D
    var tail: SN<Conv2D<Float>>
    
    var bn0: XNorm
    var bn1: XNorm
    var bn2: XNorm
    var bn3: XNorm
    var bn4: XNorm
    
    init(options: Options) {
        self.options = options
        
        head = SN(TransposedConv2D(filterShape: (4, 4, 128, options.latentSize),
                                   filterInitializer: heNormal))
        conv1 = UpSamplingConv2D(inputDim: 128, outputDim: 128, kernelSize: 4,
                                 method: options.upsampleMethod)
        conv2 = UpSamplingConv2D(inputDim: 128, outputDim: 64, kernelSize: 4,
                                 method: options.upsampleMethod)
        conv3 = UpSamplingConv2D(inputDim: 64, outputDim: 32, kernelSize: 4,
                                 method: options.upsampleMethod)
        conv4 = UpSamplingConv2D(inputDim: 32, outputDim: 16, kernelSize: 4,
                                 method: options.upsampleMethod)
        tail = SN(Conv2D(filterShape: (3, 3, 16, 3), padding: .same,
                         filterInitializer: heNormal))
        
        bn0 = XNorm(method: options.normMethod, dim: 128)
        bn1 = XNorm(method: options.normMethod, dim: 128)
        bn2 = XNorm(method: options.normMethod, dim: 64)
        bn3 = XNorm(method: options.normMethod, dim: 32)
        bn4 = XNorm(method: options.normMethod, dim: 16)
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
        
        x = lrelu(bn0(head(x))) // [-1, 4, 4, 128]
        x = lrelu(bn1(conv1(x))) // [-1, 8, 8, 128]
        x = lrelu(bn2(conv2(x))) // [-1, 16, 16, 64]
        x = lrelu(bn3(conv3(x))) // [-1, 32, 32, 32]
        x = lrelu(bn4(conv4(x))) // [-1, 64, 64, 16]
        
        x = tail(x)
        
        if options.tanhOutput {
            x = tanh(x)
        }
        
        precondition(x.shape == [input.shape[0], 64, 64, 3], "Invalid shape: \(x.shape)")
        
        return x
    }
    
    func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        writer.addHistogram(tag: "G/head.filter", values: head.filter, globalStep: globalStep)
        writer.addHistogram(tag: "G/conv1.filter", values: conv1.filter, globalStep: globalStep)
        writer.addHistogram(tag: "G/conv2.filter", values: conv2.filter, globalStep: globalStep)
        writer.addHistogram(tag: "G/conv3.filter", values: conv3.filter, globalStep: globalStep)
        writer.addHistogram(tag: "G/conv4.filter", values: conv4.filter, globalStep: globalStep)
        writer.addHistogram(tag: "G/tail.filter", values: tail.filter, globalStep: globalStep)
    }
}
