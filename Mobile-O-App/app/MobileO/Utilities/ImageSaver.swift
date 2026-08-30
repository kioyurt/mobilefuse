import Foundation
import Photos
import UIKit

/// Result of a save operation
enum SaveResult {
    case success
    case permissionDenied
    case error(Error)
}

/// Utility for saving images to the photo library
@MainActor
final class ImageSaver: ObservableObject {

    /// Saves an image to the photo library
    /// - Parameter image: The image to save
    /// - Returns: Result indicating success, permission denied, or error
    func save(_ image: PlatformImage) async -> SaveResult {
        // Request permission using addOnly (minimal permission)
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)

        guard status == .authorized || status == .limited else {
            return .permissionDenied
        }

        do {
            try await PHPhotoLibrary.shared().performChanges {
                guard let data = image.pngData() else {
                    return
                }
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: data, options: nil)
            }
            return .success
        } catch {
            return .error(error)
        }
    }

    /// Checks current photo library authorization status
    static var authorizationStatus: PHAuthorizationStatus {
        PHPhotoLibrary.authorizationStatus(for: .addOnly)
    }

    /// Opens system settings for the app (to grant photo permissions)
    static func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(url)
        }
    }
}
