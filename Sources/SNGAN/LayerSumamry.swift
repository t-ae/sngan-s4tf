import TensorFlow
import CustomLayers
import TensorBoardX

extension SN: HistogramWritable where L: HistogramWritable {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        layer.writeHistograms(tag: tag, writer: writer, globalStep: globalStep)
    }
}

extension SNDense: HistogramWritable where Scalar == Float {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        writer.addHistogram(tag: "\(tag).weight", values: dense.weight, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: dense.bias, globalStep: globalStep)
    }
}

extension SNConv2D: HistogramWritable where Scalar == Float {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        writer.addHistogram(tag: "\(tag).filter", values: conv.filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: conv.bias, globalStep: globalStep)
    }
}

extension GBlock: HistogramWritable {
    func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        writer.addHistograms(tag: "\(tag).conv1", layer: conv1, globalStep: globalStep)
        writer.addHistograms(tag: "\(tag).conv2", layer: conv2, globalStep: globalStep)
        writer.addHistograms(tag: "\(tag).convSC", layer: convSC, globalStep: globalStep)
    }
}

extension DBlock: HistogramWritable {
    func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        writer.addHistograms(tag: "\(tag).conv1", layer: conv1, globalStep: globalStep)
        writer.addHistograms(tag: "\(tag).conv2", layer: conv2, globalStep: globalStep)
        writer.addHistograms(tag: "\(tag).convSC", layer: convSC, globalStep: globalStep)
    }
}
