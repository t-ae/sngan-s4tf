import TensorFlow
import TensorBoardX

struct Generator: Layer {
    enum UpsampleMethod: String {
        case convStride, nearestNeighbor, bilinear, depthToSpace
    }
    
    @noDerivative
    var upsampleMethod: UpsampleMethod
    
    var head: SN<TransposedConv2D<Float>>
    var conv1: SN<TransposedConv2D<Float>>
    var conv2: SN<TransposedConv2D<Float>>
    var conv3: SN<TransposedConv2D<Float>>
    var conv4: SN<TransposedConv2D<Float>>
    var tail: Conv2D<Float>
    
    var bn0: Configurable<BatchNorm<Float>>
    var bn1: Configurable<BatchNorm<Float>>
    var bn2: Configurable<BatchNorm<Float>>
    var bn3: Configurable<BatchNorm<Float>>
    var bn4: Configurable<BatchNorm<Float>>
    
    init(upsampleMethod: UpsampleMethod, enableBatchNorm: Bool) {
        self.upsampleMethod = upsampleMethod
        
        let convStride = upsampleMethod == .convStride ? (2, 2) : (1, 1)
        let depthFactor = upsampleMethod == .depthToSpace ? 4 : 1
        
        head = SN(TransposedConv2D(filterShape: (4, 4, 128, latentSize),
                                   filterInitializer: heNormal))
        conv1 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 128*depthFactor, 128), strides: convStride,
                                           padding: .same, filterInitializer: heNormal))
        conv2 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 64*depthFactor, 128), strides: convStride,
                                           padding: .same, filterInitializer: heNormal))
        conv3 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 32*depthFactor, 64), strides: convStride,
                                           padding: .same, filterInitializer: heNormal))
        conv4 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 16*depthFactor, 32), strides: convStride,
                                           padding: .same, filterInitializer: heNormal))
        tail = Conv2D<Float>(filterShape: (3, 3, 16, 3), padding: .same,
                             activation: tanh, filterInitializer: heNormal)
        
        bn0 = Configurable(BatchNorm<Float>(featureCount: 128), enabled: enableBatchNorm)
        bn1 = Configurable(BatchNorm<Float>(featureCount: 128), enabled: enableBatchNorm)
        bn2 = Configurable(BatchNorm<Float>(featureCount: 64), enabled: enableBatchNorm)
        bn3 = Configurable(BatchNorm<Float>(featureCount: 32), enabled: enableBatchNorm)
        bn4 = Configurable(BatchNorm<Float>(featureCount: 16), enabled: enableBatchNorm)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = x.reshaped(to: [-1, 1, 1, latentSize])
        
        x = lrelu(bn0(head(x))) // [-1, 4, 4, 128]
        x = conv1(x)
        x = lrelu(bn1(upsample(x))) // [-1, 8, 8, 128]
        x = conv2(x)
        x = lrelu(bn2(upsample(x))) // [-1, 16, 16, 64]
        x = conv3(x)
        x = lrelu(bn3(upsample(x))) // [-1, 32, 32, 32]
        x = conv4(x)
        x = lrelu(bn4(upsample(x))) // [-1, 64, 64, 16]
        
        x = tail(x)
        
        precondition(x.shape == [input.shape[0], 64, 64, 3], "Invalid shape: \(x.shape)")
        
        return x
    }
    
    var upsampling = UpSampling2D<Float>(size: 2)
    
    @differentiable
    func upsample(_ tensor: Tensor<Float>) -> Tensor<Float> {
        switch upsampleMethod {
        case .convStride:
            return tensor
        case .nearestNeighbor:
            return upsampling(tensor)
        case .bilinear:
            return resize2xBilinear(images: tensor)
        case .depthToSpace:
            return depthToSpace(tensor, blockSize: 2)
        }
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
