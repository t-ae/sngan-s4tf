import Foundation
import TensorFlow
import GANUtils

extension Encodable {
    func prettyJsonString() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try! encoder.encode(self)
        return String(data: data, encoding: .utf8)!
    }
}

public func heNormal<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    return { shape in
        let out = shape.dimensions.dropLast().reduce(1, *)
        return Tensor(randomNormal: shape) * sqrt(2 / Scalar(out))
    }
}

public func glorotUniform<Scalar: TensorFlowFloatingPoint>(
    scale: Scalar,
    seed: TensorFlowSeed = Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    return { shape in
        Tensor(glorotUniform: shape, seed: seed) * scale
    }
}

public struct XNorm: Layer {
    public enum Method: String, Codable {
        case none, instanceNorm, batchNorm, pixelNorm
    }
    
    @noDerivative
    public var method: Method
    
    public var batchNorm: BatchNorm<Float>
    public var instanceNorm: InstanceNorm<Float>
    
    public init(method: Method, dim: Int) {
        self.method = method
        
        batchNorm = BatchNorm(featureCount: dim)
        instanceNorm = InstanceNorm(featureCount: dim)
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
            return pixelNormalization(input)
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
