import SwiftUI

/// Compact card showing an attached image thumbnail with a dismiss button.
/// Displayed above the input bar when the user picks a photo.
struct ImagePreviewCard: View {
    let image: PlatformImage
    var onDismiss: (() -> Void)? = nil

    var body: some View {
        HStack(spacing: 14) {
            ZStack(alignment: .topTrailing) {
                if image.size.width > 0 && image.size.height > 0 {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 56, height: 56)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                if let onDismiss {
                    Button(action: onDismiss) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 20))
                            .foregroundStyle(.white)
                            .background(Circle().fill(.black.opacity(0.6)).frame(width: 20, height: 20))
                    }
                    .buttonStyle(.plain)
                    .offset(x: 6, y: -6)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Image attached")
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundStyle(.primary)
                Text("Ask a question about this image")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.regularMaterial)
                .shadow(color: .black.opacity(0.04), radius: 6, y: 2)
        )
        .transition(.scale(scale: 0.96).combined(with: .opacity))
    }
}
