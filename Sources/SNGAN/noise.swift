import TensorFlow

func sampleNoise(batchSize: Int) -> Tensor<Float> {
    Tensor<Float>(randomNormal: [batchSize, latentSize])
}
