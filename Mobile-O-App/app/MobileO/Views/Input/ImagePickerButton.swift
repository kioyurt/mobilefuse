import SwiftUI
import PhotosUI
import UIKit

/// PhotosPicker wrapped as a compact icon button for the input bar.
struct ImagePickerButton: View {
    @Binding var selectedImage: PlatformImage?
    @Binding var photoPickerItem: PhotosPickerItem?
    let onReset: () -> Void

    var body: some View {
        PhotosPicker(selection: $photoPickerItem, matching: .images) {
            Image(systemName: "photo")
                .font(.system(size: 18))
                .foregroundStyle(.secondary)
                .frame(width: 36, height: 36)
        }
        .buttonStyle(.plain)
        .onChange(of: photoPickerItem) { _, newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
                        selectedImage = uiImage
                    }
                }
            }
        }
    }
}
