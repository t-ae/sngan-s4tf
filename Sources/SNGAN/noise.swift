import TensorFlow

func sampleNoise(batchSize: Int, latentSize: Int) -> Tensor<Float> {
    Tensor<Float>(randomNormal: [batchSize, latentSize])
}
