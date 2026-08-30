import SwiftUI
import UIKit

/// Circular button that saves an image to the photo library with animated state feedback.
///
/// Transitions through states: idle → saving → success/error → idle (auto-resets after 2 s).
struct SaveToGalleryButton: View {
    let image: PlatformImage

    @State private var saveState: SaveState = .idle
    @State private var showPermissionAlert = false
    @StateObject private var imageSaver = ImageSaver()

    var body: some View {
        Button {
            Task { await saveImage() }
        } label: {
            Group {
                if saveState == .saving {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: iconName)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(iconColor)
                }
            }
            .frame(width: 36, height: 36)
            .background(.regularMaterial)
            .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(saveState == .saving)
        .animation(.easeInOut(duration: 0.2), value: saveState)
        .alert("Photo Library Access Denied", isPresented: $showPermissionAlert) {
            Button("Open Settings") { ImageSaver.openSettings() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Please enable photo library access in Settings to save images.")
        }
    }

    // MARK: - State Properties

    private var iconName: String {
        switch saveState {
        case .idle, .saving: "arrow.down.to.line"
        case .success: "checkmark"
        case .error, .permissionDenied: "xmark"
        }
    }

    private var iconColor: Color {
        switch saveState {
        case .idle: .primary
        case .saving: .secondary
        case .success: .green
        case .error, .permissionDenied: .red
        }
    }

    // MARK: - Save Action

    private func saveImage() async {
        guard saveState == .idle || saveState == .error || saveState == .permissionDenied else { return }

        saveState = .saving

        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.prepare()

        let result = await imageSaver.save(image)

        switch result {
        case .success:
            saveState = .success
            generator.impactOccurred()
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        case .permissionDenied:
            saveState = .permissionDenied
            showPermissionAlert = true
        case .error:
            saveState = .error
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }

        if saveState == .success || saveState == .error {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            saveState = .idle
        }
    }
}

// MARK: - Save State

private enum SaveState {
    case idle, saving, success, error, permissionDenied
}
