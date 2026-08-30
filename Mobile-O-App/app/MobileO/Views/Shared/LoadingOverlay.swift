import SwiftUI

/// Full-screen blurred overlay with a spinner, shown while models are loading.
struct LoadingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.001)
                .background(.ultraThinMaterial)
                .ignoresSafeArea()
                .transition(.opacity)

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .progressViewStyle(.circular)
                    .tint(.white)

                Text("Loading models...")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
            }
            .transition(.scale(scale: 0.9).combined(with: .opacity))
        }
    }
}
