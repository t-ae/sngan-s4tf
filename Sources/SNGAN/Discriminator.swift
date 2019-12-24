import TensorFlow
import TensorBoardX

struct Discriminator: Layer {
    struct Options: Codable {
        var downsampleMethod: DownSamplingConv2D.Method
        var enableSpectralNorm: Bool
        var enableBatchNorm: Bool
        var enableMinibatchStdConcat: Bool
    }
    
    @noDerivative
    var options: Options
    
    var head: SN<Conv2D<Float>>
    var conv1: DownSamplingConv2D
    var conv2: DownSamplingConv2D
    var conv3: DownSamplingConv2D
    var conv4: DownSamplingConv2D
    var tail: SN<Conv2D<Float>>
    
    var stdConcat: MinibatchStdConcat
    
    var bn0: Configurable<BatchNorm<Float>>
    var bn1: Configurable<BatchNorm<Float>>
    var bn2: Configurable<BatchNorm<Float>>
    var bn3: Configurable<BatchNorm<Float>>
    var bn4: Configurable<BatchNorm<Float>>
    
    init(options: Options) {
        self.options = options
        
        head = SN(Conv2D(filterShape: (1, 1, 3, 16), filterInitializer: heNormal))
        conv1 = DownSamplingConv2D(inputDIm: 16, outputDim: 32, kernelSize: 4,
                                   method: options.downsampleMethod)
        conv2 = DownSamplingConv2D(inputDIm: 32, outputDim: 64, kernelSize: 4,
                                   method: options.downsampleMethod)
        conv3 = DownSamplingConv2D(inputDIm: 64, outputDim: 128, kernelSize: 4,
                                   method: options.downsampleMethod)
        conv4 = DownSamplingConv2D(inputDIm: 128, outputDim: 128, kernelSize: 4,
                                   method: options.downsampleMethod)
        
        let stdDim = options.enableMinibatchStdConcat ? 1 : 0
        stdConcat = MinibatchStdConcat(groupSize: 4)
        tail = SN(Conv2D(filterShape: (4, 4, 128 + stdDim, 1), filterInitializer: heNormal))
        
        let enableBatchNorm = options.enableBatchNorm
        bn0 = Configurable(BatchNorm<Float>(featureCount: 16), enabled: enableBatchNorm)
        bn1 = Configurable(BatchNorm<Float>(featureCount: 32), enabled: enableBatchNorm)
        bn2 = Configurable(BatchNorm<Float>(featureCount: 64), enabled: enableBatchNorm)
        bn3 = Configurable(BatchNorm<Float>(featureCount: 128), enabled: enableBatchNorm)
        bn4 = Configurable(BatchNorm<Float>(featureCount: 128), enabled: enableBatchNorm)
    }
    
    mutating func preTrain() {
        if(options.enableSpectralNorm) {
            spectralNormalize(&self)
        }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = lrelu(bn0(head(x))) // [-1, 64, 64, 16]
        x = lrelu(bn1(conv1(x))) // [-1, 32, 32, 32]
        x = lrelu(bn2(conv2(x))) // [-1, 16, 16, 64]
        x = lrelu(bn3(conv3(x))) // [-1, 8, 8, 128]
        x = lrelu(bn4(conv4(x))) // [-1, 4, 4, 128]
        if options.enableMinibatchStdConcat {
            x = stdConcat(x)  // [-1, 4, 4, 129]
        }
        x = tail(x) // [-1, 1, 1, 1]
        x = x.squeezingShape(at: [1, 2])
        
        return x
    }

    func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        writer.addHistogram(tag: "D/head.filter", values: head.filter, globalStep: globalStep)
        writer.addHistogram(tag: "D/conv1.filter", values: conv1.filter, globalStep: globalStep)
        writer.addHistogram(tag: "D/conv2.filter", values: conv2.filter, globalStep: globalStep)
        writer.addHistogram(tag: "D/conv3.filter", values: conv3.filter, globalStep: globalStep)
        writer.addHistogram(tag: "D/conv4.filter", values: conv3.filter, globalStep: globalStep)
        writer.addHistogram(tag: "D/tail.filter", values: tail.filter, globalStep: globalStep)
    }
}
