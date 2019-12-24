import TensorFlow

// MARK: - Upsampling conv
struct UpSamplingConv2D: Layer {
    enum Method: String, Codable {
        case convStride, nearestNeighbor, bilinear, depthToSpace
    }
    
    @noDerivative
    var method: Method
    var conv: SN<TransposedConv2D<Float>>
    
    init(inputDim: Int, outputDim: Int, kernelSize: Int, method: Method) {
        self.method = method
        let depthFactor = method == .depthToSpace ? 4 : 1
        let strides = method == .convStride ? (2, 2) : (1, 1)
        let filterShape = (kernelSize, kernelSize, outputDim*depthFactor, inputDim)
        conv = SN(TransposedConv2D(filterShape: filterShape, strides: strides,
                                   padding: .same, filterInitializer: heNormal))
    }
    
    var upsampling = UpSampling2D<Float>(size: 2)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = conv(input)
        
        switch method {
        case .convStride:
            break
        case .nearestNeighbor:
            x = upsampling(x)
        case .bilinear:
            x = resize2xBilinear(images: x)
        case .depthToSpace:
            x = depthToSpace(x, blockSize: 2)
        }
        
        return x
    }
    
    var filter: Tensor<Float> {
        conv.filter
    }
}

// MARK: - Downsampling conv
struct DownSamplingConv2D: Layer {
    enum Method: String, Codable {
        case convStride, avgPool
    }
    
    @noDerivative
    var method: Method
    var conv: SN<Conv2D<Float>>
    
    init(inputDIm: Int, outputDim: Int, kernelSize: Int, method: Method) {
        self.method = method
        let strides = method == .convStride ? (2, 2) : (1, 1)
        conv = SN(Conv2D(filterShape: (kernelSize, kernelSize, inputDIm, outputDim), strides: strides,
                         padding: .same, filterInitializer: heNormal))
    }
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = conv(input)
        
        switch method {
        case .convStride:
            break
        case .avgPool:
            x = avgPool(x)
        }
        
        return x
    }
    
    var filter: Tensor<Float> {
        conv.filter
    }
}

// MARK: - Normalization selector
struct XNorm: Layer {
    enum Method: String, Codable {
        case none, instanceNorm, batchNorm, pixelNorm
    }
    
    @noDerivative
    var method: Method
    
    var batchNorm: BatchNorm<Float>
    var instanceNorm: InstanceNorm2D<Float>
    var pixelNorm: PixelNorm2D<Float>
    
    init(method: Method, dim: Int) {
        self.method = method
        
        batchNorm = BatchNorm(featureCount: dim)
        instanceNorm = InstanceNorm2D()
        pixelNorm = PixelNorm2D()
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch method {
        case .none:
            return input
        case .instanceNorm:
            return instanceNorm(input)
        case .batchNorm:
            return batchNorm(input)
        case .pixelNorm:
            return pixelNorm(input)
        }
    }
}
