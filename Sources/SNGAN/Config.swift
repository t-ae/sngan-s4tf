import Foundation

struct Config: Codable {
    let batchSize: Int
    let nDisUpdate: Int
    let loss: GANLossType
}
