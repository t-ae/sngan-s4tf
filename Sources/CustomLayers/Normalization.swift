import TensorFlow

public struct ConditionalBatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    public struct Input: Differentiable {
        public var feature: Tensor<Scalar>
        @noDerivative
        public var label: Tensor<Int32>
        
        @differentiable
        public init(feature: Tensor<Scalar>, label: Tensor<Int32>) {
            self.feature = feature
            self.label = label
        }
    }
    
    @noDerivative
    public let featureCount: Int
    
    public var bn: BatchNorm<Scalar>
    
    public var gammaEmb: Embedding<Scalar>
    public var betaEmb: Embedding<Scalar>
    
    public init(featureCount: Int) {
        self.featureCount = featureCount
        self.bn = BatchNorm(featureCount: featureCount)
        self.gammaEmb = Embedding(embeddings: Tensor<Scalar>(ones: [10, featureCount]))
        self.betaEmb = Embedding(embeddings: Tensor<Scalar>(zeros: [10, featureCount]))
    }
    
    @differentiable
    public func callAsFunction(_ input: Input) -> Tensor<Scalar> {
        let x = bn(input.feature)
        
        let gamma = gammaEmb(input.label).reshaped(to: [-1, 1, 1, featureCount])
        let beta = betaEmb(input.label).reshaped(to: [-1, 1, 1, featureCount])
        
        return x  * gamma + beta
    }
}

// https://arxiv.org/abs/1607.08022
public struct InstanceNorm2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let mean = input.mean(alongAxes: 1, 2)
        let variance = squaredDifference(input, mean).mean(alongAxes: 1, 2)
        return input * rsqrt(variance + 1e-8)
    }
}

// https://arxiv.org/abs/1710.10196
public struct PixelNorm2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let sqmean = input.squared().mean(alongAxes: 3)
        return input * rsqrt(sqmean + 1e-8)
    }
}

