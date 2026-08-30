import SwiftUI

/// Empty-state welcome screen shown when the conversation has no messages.
/// Displays the app logo, capability badges, and a 2×2 grid of example prompts.
struct WelcomeView: View {
    @Binding var prompt: String
    @Binding var showImagePicker: Bool
    @Binding var selectedImage: PlatformImage?
    var hasImage: Bool

    var body: some View {
        VStack(spacing: 28) {
            header
            howItWorks
            exampleGrid
        }
        .frame(maxWidth: 520)
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .transition(.scale(scale: 0.94).combined(with: .opacity))
    }

    // MARK: - Header

    private var header: some View {
        VStack(spacing: 12) {
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 48))
                .foregroundStyle(.purple.opacity(0.7))
                .padding(.top, 20)

            Text("Welcome to Mobile-O")
                .font(.system(size: 26, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            capabilityBadges
        }
        .padding(.bottom, 8)
    }

    private var capabilityBadges: some View {
        HStack(spacing: 8) {
            CapabilityBadge(icon: "sparkles", label: "Create", color: .purple)
            badgeSeparator
            CapabilityBadge(icon: "photo", label: "Analyze", color: .blue)
            badgeSeparator
            CapabilityBadge(icon: "bubble.left", label: "Chat", color: .green)
        }
    }

    private var badgeSeparator: some View {
        Text("•")
            .font(.system(size: 10))
            .foregroundStyle(.secondary.opacity(0.5))
    }

    // MARK: - How It Works

    private var howItWorks: some View {
        VStack(spacing: 8) {
            HowItWorksRow(
                icon: "sparkles", color: .purple,
                title: "Create Images",
                subtitle: "Tap \u{2726} and describe what you want",
                onTap: { prompt = "Generate " }
            )
            HowItWorksRow(
                icon: "photo", color: .blue,
                title: "Analyze Photos",
                subtitle: "Attach a photo and ask about it"
            )
            HowItWorksRow(
                icon: "bubble.left", color: .green,
                title: "Chat",
                subtitle: "Just type to start a conversation"
            )
        }
        .frame(maxWidth: 520)
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.regularMaterial)
                .shadow(color: .black.opacity(0.05), radius: 8, y: 3)
        )
    }

    // MARK: - Example Grid

    private var exampleGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Try an example")
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 4)

            LazyVGrid(columns: [
                GridItem(.flexible(), spacing: 10),
                GridItem(.flexible(), spacing: 10)
            ], spacing: 10) {
                ExamplePromptChip(
                    icon: "sparkles", color: .purple,
                    displayText: "Generate a flower",
                    onTap: { prompt = "Generate a red rose with morning dew drops" }
                )
                ExamplePromptChip(
                    icon: "sparkles", color: .purple,
                    displayText: "Generate landscape",
                    onTap: { prompt = "Generate a serene mountain lake surrounded by autumn trees" }
                )
                ExamplePromptChip(
                    icon: "photo", color: .blue,
                    displayText: "Describe image",
                    caption: hasImage ? nil : "Tap to attach",
                    onTap: {
                        prompt = "What's in this image? Describe it in detail"
                        if !hasImage { showImagePicker = true }
                    }
                )
                ExamplePromptChip(
                    icon: "bubble.left", color: .green,
                    displayText: "Explain topic",
                    onTap: { prompt = "Explain quantum computing in simple terms that anyone can understand" }
                )
            }
        }
        .frame(maxWidth: 520)
        .padding(.horizontal, 16)
        .padding(.vertical, 16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.regularMaterial)
                .shadow(color: .black.opacity(0.05), radius: 8, y: 3)
        )
    }
}

// MARK: - Capability Badge

private struct CapabilityBadge: View {
    let icon: String
    let label: String
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - How It Works Row

private struct HowItWorksRow: View {
    let icon: String
    let color: Color
    let title: String
    let subtitle: String
    var onTap: (() -> Void)? = nil

    var body: some View {
        let content = HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(color)
                .frame(width: 32, height: 32)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(color.opacity(0.1))
                )

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(subtitle)
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            if onTap != nil {
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(color.opacity(0.04))
        )

        if let onTap {
            Button(action: onTap) { content }
                .buttonStyle(.plain)
        } else {
            content
        }
    }
}

// MARK: - Example Prompt Chip

private struct ExamplePromptChip: View {
    let icon: String
    let color: Color
    let displayText: String
    var caption: String? = nil
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 24, weight: .medium))
                    .foregroundStyle(color)
                    .frame(height: 28)

                VStack(spacing: 2) {
                    Text(displayText)
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)

                    if let caption {
                        Text(caption)
                            .font(.system(size: 9, design: .rounded))
                            .foregroundStyle(.secondary)
                    } else {
                        Text(" ")
                            .font(.system(size: 9, design: .rounded))
                            .opacity(0)
                    }
                }
            }
            .frame(maxWidth: .infinity, minHeight: 100)
            .padding(.horizontal, 12)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(color.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(color.opacity(0.2), lineWidth: 1.5)
            )
        }
        .buttonStyle(.plain)
    }
}
