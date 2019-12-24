import TensorFlow

struct ConditionalBatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    struct Input: Differentiable {
        var feature: Tensor<Scalar>
        @noDerivative
        var label: Tensor<Int32>
        
        @differentiable
        init(feature: Tensor<Scalar>, label: Tensor<Int32>) {
            self.feature = feature
            self.label = label
        }
    }
    
    @noDerivative
    let featureCount: Int
    
    var bn: BatchNorm<Scalar>
    
    var gammaEmb: Embedding<Scalar>
    var betaEmb: Embedding<Scalar>
    
    init(featureCount: Int) {
        self.featureCount = featureCount
        self.bn = BatchNorm(featureCount: featureCount)
        self.gammaEmb = Embedding(embeddings: Tensor<Scalar>(ones: [10, featureCount]))
        self.betaEmb = Embedding(embeddings: Tensor<Scalar>(zeros: [10, featureCount]))
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Tensor<Scalar> {
        let x = bn(input.feature)
        
        let gamma = gammaEmb(input.label).reshaped(to: [-1, 1, 1, featureCount])
        let beta = betaEmb(input.label).reshaped(to: [-1, 1, 1, featureCount])
        
        return x  * gamma + beta
    }
}

// https://arxiv.org/abs/1607.08022
struct InstanceNorm2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let mean = input.mean(alongAxes: 1, 2)
        let variance = squaredDifference(input, mean).mean(alongAxes: 1, 2)
        return input * rsqrt(variance + 1e-8)
    }
}

// https://arxiv.org/abs/1710.10196
struct PixelNorm2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let sqmean = input.squared().mean(alongAxes: 3)
        return input * rsqrt(sqmean + 1e-8)
    }
}

