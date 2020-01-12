import TensorFlow

func sampleNoise(batchSize: Int, latentSize: Int) -> Tensor<Float> {
    Tensor<Float>(randomNormal: [batchSize, latentSize])
}

func sampleInterpolationNoise(latentSize: Int) -> Tensor<Float> {
    let tensor1 = sampleNoise(batchSize: 8, latentSize: latentSize).expandingShape(at: 1)
    let tensor2 = sampleNoise(batchSize: 8, latentSize: latentSize).expandingShape(at: 1)
    
    let rates = (0..<8).map { Float($0) / 8 }
    
    return Tensor(concatenating: rates.map { rate in
        tensor1 - rate * (tensor1 - tensor2)
        }, alongAxis: 1)
        .reshaped(to: [-1, latentSize])
}
