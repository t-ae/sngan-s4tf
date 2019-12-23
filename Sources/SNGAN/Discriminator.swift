import TensorFlow
import TensorBoardX

struct Discriminator: Layer {
    var head = SN(Conv2D(filterShape: (1, 1, 3, 16), filterInitializer: heNormal))
    var conv1 = SN(Conv2D(filterShape: (4, 4, 16, 32), strides: (2, 2),
                          padding: .same, filterInitializer: heNormal))
    var conv2 = SN(Conv2D(filterShape: (4, 4, 32, 64), strides: (2, 2),
                          padding: .same, filterInitializer: heNormal))
    var conv3 = SN(Conv2D(filterShape: (4, 4, 64, 128), strides: (2, 2),
                          padding: .same, filterInitializer: heNormal))
    var conv4 = SN(Conv2D(filterShape: (4, 4, 128, 128), strides: (2, 2),
                          padding: .same, filterInitializer: heNormal))
    var tail = SN(Conv2D(filterShape: (4, 4, 128, 1), filterInitializer: heNormal))
    
    var bn0 = BatchNorm<Float>(featureCount: 16)
    var bn1 = BatchNorm<Float>(featureCount: 32)
    var bn2 = BatchNorm<Float>(featureCount: 64)
    var bn3 = BatchNorm<Float>(featureCount: 128)
    var bn4 = BatchNorm<Float>(featureCount: 128)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        if enableBatchNormalization.D {
            x = lrelu(bn0(head(x))) // [-1, 64, 64, 16]
            x = lrelu(bn1(conv1(x))) // [-1, 32, 32, 32]
            x = lrelu(bn2(conv2(x))) // [-1, 16, 16, 64]
            x = lrelu(bn3(conv3(x))) // [-1, 8, 8, 128]
            x = lrelu(bn4(conv4(x))) // [-1, 8, 8, 128]
        } else {
            x = lrelu(head(x)) // [-1, 64, 64, 16]
            x = lrelu(conv1(x)) // [-1, 32, 32, 32]
            x = lrelu(conv2(x)) // [-1, 16, 16, 64]
            x = lrelu(conv3(x)) // [-1, 8, 8, 128]
            x = lrelu(conv4(x)) // [-1, 4, 4, 128]
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
