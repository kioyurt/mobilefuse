import UIKit

/// Platform-specific image type (UIImage on iOS).
public typealias PlatformImage = UIImage

extension PlatformImage {
    /// Identity-based comparison (pointer equality) for fast SwiftUI diffing.
    func isEqual(_ other: PlatformImage) -> Bool {
        self === other
    }
}
