import TensorFlow
import TensorBoardX

struct Generator: Layer {
    var head = SN(TransposedConv2D(filterShape: (4, 4, 128, latentSize),
                                   filterInitializer: heNormal))
    var conv1 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 128, 128), strides: (2, 2),
                                           padding: .same, filterInitializer: heNormal))
    var conv2 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 64, 128), strides: (2, 2),
                                           padding: .same, filterInitializer: heNormal))
    var conv3 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 32, 64), strides: (2, 2),
                                           padding: .same, filterInitializer: heNormal))
    var conv4 = SN(TransposedConv2D<Float>(filterShape: (4, 4, 16, 32), strides: (2, 2),
                                           padding: .same, filterInitializer: heNormal))
    
    var bn0 = BatchNorm<Float>(featureCount: 128)
    var bn1 = BatchNorm<Float>(featureCount: 128)
    var bn2 = BatchNorm<Float>(featureCount: 64)
    var bn3 = BatchNorm<Float>(featureCount: 32)
    var bn4 = BatchNorm<Float>(featureCount: 16)
    
    var tail = Conv2D<Float>(filterShape: (3, 3, 16, 3), padding: .same,
                             activation: tanh, filterInitializer: heNormal)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = x.reshaped(to: [-1, 1, 1, latentSize])
        if enableBatchNormalization.G {
            x = lrelu(bn0(head(x))) // [-1, 4, 4, 128]
            x = lrelu(bn1(conv1(x))) // [-1, 8, 8, 128]
            x = lrelu(bn2(conv2(x))) // [-1, 16, 16, 64]
            x = lrelu(bn3(conv3(x))) // [-1, 32, 32, 32]
            x = lrelu(bn4(conv4(x))) // [-1, 64, 64, 16]
        } else {
            x = lrelu(head(x)) // [-1, 4, 4, 128]
            x = lrelu(conv1(x)) // [-1, 8, 8, 128]
            x = lrelu(conv2(x)) // [-1, 16, 16, 64]
            x = lrelu(conv3(x)) // [-1, 32, 32, 32]
            x = lrelu(conv4(x)) // [-1, 64, 64, 16]
        }
        x = tail(x)
        
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
