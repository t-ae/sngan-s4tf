import TensorFlow
import CustomLayers
import TensorBoardX

extension Dense where Scalar == Float {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).weight", values: weight, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: bias, globalStep: globalStep)
    }
}

extension Conv2D where Scalar == Float {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).filter", values: filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: bias, globalStep: globalStep)
    }
}

extension TransposedConv2D where Scalar == Float {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).filter", values: filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: bias, globalStep: globalStep)
    }
}

extension SN where L == Conv2D<Float> {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).filter", values: filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: bias, globalStep: globalStep)
    }
}

extension SN where L == TransposedConv2D<Float> {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).filter", values: filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: bias, globalStep: globalStep)
    }
}

extension SNDense where Scalar == Float {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).weight", values: dense.weight, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: dense.bias, globalStep: globalStep)
    }
}

extension SNConv2D where Scalar == Float {
    func writeHistograms(writer: SummaryWriter, tag: String, globalStep: Int) {
        writer.addHistogram(tag: "\(tag).filter", values: conv.filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: conv.bias, globalStep: globalStep)
    }
}
