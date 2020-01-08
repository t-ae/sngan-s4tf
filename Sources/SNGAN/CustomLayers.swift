import TensorFlow
import TensorBoardX
import CustomLayers

// MARK: - Upsampling conv
struct UpSamplingConv2D: Layer {
    enum Method: String, Codable {
        case nearestNeighbor, bilinear, depthToSpace
    }
    
    @noDerivative
    var method: Method
    var conv: SNConv2D<Float>
    
    init(inputDim: Int,
         outputDim: Int,
         kernelSize: Int,
         method: Method,
         enableSpectralNorm: Bool) {
        self.method = method
        let depthFactor = method == .depthToSpace ? 4 : 1
        let filterShape = (kernelSize, kernelSize, inputDim, outputDim*depthFactor)
        conv = SNConv2D(Conv2D(filterShape: filterShape, strides: (1, 1),
                               padding: .same, filterInitializer: heNormal()),
                        enabled: enableSpectralNorm)
    }
    
    var upsampling = UpSampling2D<Float>(size: 2)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = conv(input)
        
        switch method {
        case .nearestNeighbor:
            x = upsampling(x)
        case .bilinear:
            x = resize2xBilinear(images: x)
        case .depthToSpace:
            x = depthToSpace(x, blockSize: 2)
        }
        
        return x
    }
}

// MARK: - Downsampling conv
struct DownSamplingConv2D: Layer {
    enum Method: String, Codable {
        case convStride, avgPool
    }
    
    @noDerivative
    var method: Method
    var conv: SNConv2D<Float>
    
    init(inputDIm: Int,
         outputDim: Int,
         kernelSize: Int,
         method: Method,
         enableSpectralNorm: Bool) {
        self.method = method
        let strides = method == .convStride ? (2, 2) : (1, 1)
        conv = SNConv2D(Conv2D(filterShape: (kernelSize, kernelSize, inputDIm, outputDim),
                               strides: strides, padding: .same, filterInitializer: heNormal()),
                        enabled: enableSpectralNorm)
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

struct Activation: ParameterlessLayer {
    enum Method: String, Codable {
        case relu, leakyRelu, elu
    }
    
    let method: Method
    
    init(method: Method) {
        self.method = method
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch method {
        case .relu:
            return relu(input)
        case .leakyRelu:
            return leakyRelu(input)
        case .elu:
            return elu(input)
        }
    }
}

struct Resize: ParameterlessLayer {
    enum Method: String, Codable {
        case nearestNeighbor, bilinear
    }
    
    @noDerivative
    let width: Int
    @noDerivative
    let height: Int
    @noDerivative
    let method: Method
    @noDerivative
    let alignCorners: Bool
    
    init(width: Int, height: Int, method: Method, alignCorners: Bool) {
        self.width = width
        self.height = height
        self.method = method
        self.alignCorners = alignCorners
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch method {
        case .nearestNeighbor:
            return resizeNN(images: input, width: width, height: height, alignCorners: alignCorners)
        case .bilinear:
            return resizeBL(images: input, width: width, height: height, alignCorners: alignCorners)
        }
    }
}
