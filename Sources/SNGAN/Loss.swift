import TensorFlow

enum GANLossType: String, Codable {
    case nonSaturating, lsgan, hinge
}

struct GANLoss {
    let type: GANLossType
    
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .nonSaturating:
            return softplus(-tensor).mean()
        case .lsgan:
            return pow(tensor - 1, 2).mean()
        case .hinge:
            return -tensor.mean()
        }
    }
    
    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .nonSaturating:
            return softplus(-real).mean() + softplus(fake).mean()
        case .lsgan:
            return pow(real-1, 2).mean() + pow(fake, 2).mean()
        case .hinge:
            return relu(1 - real).mean() + relu(1 + fake).mean()
        }
    }
}
