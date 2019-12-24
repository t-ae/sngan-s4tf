import TensorFlow
import TensorBoardX

struct Discriminator: Layer {
    struct Options: Codable {
        var downsampleMethod: DownsampleMethod
        var enableSpectralNorm: Bool
        var enableBatchNorm: Bool
        var enableMinibatchStdConcat: Bool
    }
    
    enum DownsampleMethod: String, Codable {
        case convStride, avgPool
    }
    
    @noDerivative
    var options: Options
    
    var head: SN<Conv2D<Float>>
    var conv1: SN<Conv2D<Float>>
    var conv2: SN<Conv2D<Float>>
    var conv3: SN<Conv2D<Float>>
    var conv4: SN<Conv2D<Float>>
    var tail: SN<Conv2D<Float>>
    
    var stdConcat: MinibatchStdConcat
    
    var bn0: Configurable<BatchNorm<Float>>
    var bn1: Configurable<BatchNorm<Float>>
    var bn2: Configurable<BatchNorm<Float>>
    var bn3: Configurable<BatchNorm<Float>>
    var bn4: Configurable<BatchNorm<Float>>
    
    init(options: Options) {
        self.options = options
        
        let convStride = options.downsampleMethod == .convStride ? (2, 2) : (1, 1)
        head = SN(Conv2D(filterShape: (1, 1, 3, 16), filterInitializer: heNormal))
        conv1 = SN(Conv2D(filterShape: (4, 4, 16, 32), strides: convStride,
                          padding: .same, filterInitializer: heNormal))
        conv2 = SN(Conv2D(filterShape: (4, 4, 32, 64), strides: convStride,
                          padding: .same, filterInitializer: heNormal))
        conv3 = SN(Conv2D(filterShape: (4, 4, 64, 128), strides: convStride,
                          padding: .same, filterInitializer: heNormal))
        conv4 = SN(Conv2D(filterShape: (4, 4, 128, 128), strides: convStride,
                          padding: .same, filterInitializer: heNormal))
        
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
        x = lrelu(bn1(conv1(x)))
        x = downsample(x) // [-1, 32, 32, 32]
        x = lrelu(bn2(conv2(x)))
        x = downsample(x) // [-1, 16, 16, 64]
        x = lrelu(bn3(conv3(x)))
        x = downsample(x) // [-1, 8, 8, 128]
        x = lrelu(bn4(conv4(x)))
        x = downsample(x) // [-1, 4, 4, 128]
        if options.enableMinibatchStdConcat {
            x = stdConcat(x)  // [-1, 4, 4, 129]
        }
        x = tail(x) // [-1, 1, 1, 1]
        x = x.squeezingShape(at: [1, 2])
        
        return x
    }
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @differentiable
    func downsample(_ tensor: Tensor<Float>) -> Tensor<Float> {
        switch options.downsampleMethod {
        case .convStride:
            return tensor
        case .avgPool:
            return avgPool(tensor)
        }
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
