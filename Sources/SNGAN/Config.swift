import Foundation
import GANUtils

struct Config: Codable {
    let batchSize: Int
    let nDisUpdate: Int
    let loss: GANLossType
    let imageSize: ImageSize
}

enum ImageSize: Int, Comparable, Codable {
    case x32 = 32
    case x64 = 64
    case x128 = 128
    
    static func <(lhs: ImageSize, rhs: ImageSize) -> Bool {
        switch (lhs, rhs) {
        case (.x32, x64), (.x32, .x128), (.x64, .x128):
            return true
        default:
            return false
        }
    }
    
    var plotGridCols: Int {
        switch self {
        case .x32: return 16
        case .x64: return 8
        case .x128: return 6
        }
    }
}
