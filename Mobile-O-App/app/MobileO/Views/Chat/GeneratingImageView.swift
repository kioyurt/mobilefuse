import SwiftUI

/// 480x480 progress card displayed in the conversation while image generation is running.
/// Shows an animated sparkle icon inside a circular progress ring, plus step/percentage pills.
struct GeneratingImageView: View {
    let currentStep: Int
    let totalSteps: Int

    @State private var iconPulse = false
    @State private var progressOpacity = 0.0

    private var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(currentStep) / Double(totalSteps)
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
                .frame(width: 480, height: 480)

            VStack(spacing: 24) {
                progressRing
                progressInfo
                    .opacity(progressOpacity)
            }
        }
        .frame(width: 480, height: 480)
        .background(Color.black.opacity(0.02))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.1), radius: 16, y: 4)
        .transition(.asymmetric(
            insertion: .scale(scale: 0.96).combined(with: .opacity),
            removal: .identity
        ))
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.5).repeatForever(autoreverses: true)) {
                iconPulse = true
            }
            withAnimation(.easeInOut(duration: 0.5)) {
                progressOpacity = 1.0
            }
        }
    }

    // MARK: - Subviews

    /// Animated sparkle icon inside a circular progress ring.
    private var progressRing: some View {
        ZStack {
            Circle()
                .stroke(Color.purple.opacity(0.2), lineWidth: 4)
                .frame(width: 80, height: 80)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    LinearGradient(
                        colors: [.purple, .blue],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    style: StrokeStyle(lineWidth: 4, lineCap: .round)
                )
                .frame(width: 80, height: 80)
                .rotationEffect(.degrees(-90))
                .animation(.easeInOut(duration: 0.3), value: progress)

            Circle()
                .fill(.ultraThinMaterial)
                .frame(width: 64, height: 64)

            Image(systemName: "sparkles")
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(.purple)
                .scaleEffect(iconPulse ? 1.1 : 1.0)
                .rotationEffect(.degrees(iconPulse ? 3 : -3))
        }
    }

    /// Step counter and percentage pills below the ring.
    private var progressInfo: some View {
        VStack(spacing: 8) {
            Text("Generating image...")
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary)

            HStack(spacing: 12) {
                pill(icon: "number", text: "Step \(currentStep) of \(totalSteps)", color: .secondary, bgOpacity: 0.05)
                pill(icon: "chart.bar.fill", text: "\(Int(progress * 100))%", color: .purple, bgOpacity: 0.1)
            }
        }
    }

    private func pill(icon: String, text: String, color: Color, bgOpacity: Double) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
            Text(text)
                .font(.system(size: 13, weight: .medium, design: .rounded))
        }
        .foregroundStyle(color)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(color.opacity(bgOpacity))
        .clipShape(Capsule())
    }
}
