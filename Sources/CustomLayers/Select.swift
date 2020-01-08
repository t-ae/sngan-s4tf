import TensorFlow

public struct XNorm: Layer {
    public enum Method: String, Codable {
        case none, instanceNorm, batchNorm, pixelNorm
    }
    
    @noDerivative
    public var method: Method
    
    public var batchNorm: BatchNorm<Float>
    public var instanceNorm: InstanceNorm2D<Float>
    public var pixelNorm: PixelNorm2D<Float>
    
    public init(method: Method, dim: Int) {
        self.method = method
        
        batchNorm = BatchNorm(featureCount: dim)
        instanceNorm = InstanceNorm2D()
        pixelNorm = PixelNorm2D()
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
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

public struct Activation: ParameterlessLayer {
    public enum Method: String, Codable {
        case relu, leakyRelu, elu
    }
    
    public let method: Method
    
    public init(method: Method) {
        self.method = method
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
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

public struct Resize: ParameterlessLayer {
    public enum Method: String, Codable {
        case nearestNeighbor, bilinear
    }
    
    @noDerivative
    public let width: Int
    @noDerivative
    public let height: Int
    @noDerivative
    public let method: Method
    @noDerivative
    public let alignCorners: Bool
    
    public init(width: Int, height: Int, method: Method, alignCorners: Bool) {
        self.width = width
        self.height = height
        self.method = method
        self.alignCorners = alignCorners
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch method {
        case .nearestNeighbor:
            return resizeNN(images: input, width: width, height: height, alignCorners: alignCorners)
        case .bilinear:
            return resizeBL(images: input, width: width, height: height, alignCorners: alignCorners)
        }
    }
}
