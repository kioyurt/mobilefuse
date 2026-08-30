import SwiftUI
import UIKit

/// Renders a single chat message as a styled bubble.
///
/// Supports three content types:
/// - `.text`           → plain text (or streaming text with a blinking cursor)
/// - `.image`          → generated image with context-menu save/share
/// - `.textWithImage`  → user prompt with an attached photo
///
struct MessageBubbleView: View {
    let message: ChatMessage
    var isStreaming: Bool = false

    private var isFromUser: Bool { message.role == .user }

    private var isImageMessage: Bool {
        if case .image = message.content { return true }
        return false
    }

    private var isInterruption: Bool {
        if case .text(let t) = message.content { return t.contains("generation interrupted") }
        return false
    }

    // MARK: - Body

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            if isFromUser { Spacer(minLength: 80) }

            VStack(alignment: isFromUser ? .trailing : .leading, spacing: 6) {
                contentView
                metadataRow
            }
            .frame(maxWidth: 600, alignment: isFromUser ? .trailing : .leading)
            .fixedSize(horizontal: false, vertical: true)

            if !isFromUser { Spacer(minLength: 80) }
        }
        .padding(.horizontal, 16)
    }

    // MARK: - Content Routing

    @ViewBuilder
    private var contentView: some View {
        switch message.content {
        case .text(let text):
            textBubble(text)
        case .image(let image):
            imageBubble(image)
        case .textWithImage(let text, let image):
            textWithImageBubble(text: text, image: image)
        }
    }

    // MARK: - Text Bubble

    private func textBubble(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if isInterruption {
                interruptionBanner
            } else if isStreaming && !isFromUser {
                streamingTextView(text)
            } else {
                staticTextView(text)
            }

            if isStreaming {
                Divider()
                    .padding(.vertical, 10)
                    .transition(.opacity.animation(.easeInOut(duration: 0.2)))
                streamingIndicator
                    .transition(.opacity.animation(.easeInOut(duration: 0.2)))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(textBubbleBackground)
    }

    private var interruptionBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.orange)
            Text("Image Generation Interrupted")
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary)
        }
    }

    private func streamingTextView(_ text: String) -> some View {
        StreamingText(fullText: text)
            .font(.system(size: 16))
            .foregroundStyle(.primary)
            .lineSpacing(5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
            .layoutPriority(1)
    }

    private func staticTextView(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 16))
            .foregroundStyle(isFromUser ? .white : .primary)
            .textSelection(.enabled)
            .lineSpacing(5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
            .layoutPriority(1)
    }

    private var textBubbleBackground: some View {
        ZStack {
            if isInterruption {
                Color.orange.opacity(0.12)
            } else if isFromUser {
                LinearGradient(
                    colors: [.accentColor, .accentColor.opacity(0.85)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            } else {
                Color(uiColor: .secondarySystemGroupedBackground)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(bubbleBorderColor, lineWidth: 1)
        )
        .shadow(color: .black.opacity(isFromUser ? 0.15 : 0.06), radius: isFromUser ? 8 : 4, y: 2)
    }

    private var bubbleBorderColor: Color {
        if isFromUser { return .clear }
        if isInterruption { return .orange.opacity(0.3) }
        return .primary.opacity(0.06)
    }

    // MARK: - Image Bubble (generated output)

    private func imageBubble(_ image: PlatformImage) -> some View {
        HStack(alignment: .center, spacing: 12) {
            GeneratedImageView(image: image)
            SaveToGalleryButton(image: image)
        }
        .transition(.asymmetric(
            insertion: .scale(scale: 0.96).combined(with: .opacity),
            removal: .identity
        ))
    }

    // MARK: - Text + Image Bubble (user attachment)

    private func textWithImageBubble(text: String, image: PlatformImage) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if image.size.width > 0 && image.size.height > 0 {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(image.size.width / image.size.height, contentMode: .fit)
                    .frame(maxWidth: 320)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            if !text.isEmpty {
                Text(text)
                    .font(.system(size: 16))
                    .foregroundStyle(.white)
                    .textSelection(.enabled)
                    .lineSpacing(5)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(14)
        .background(
            LinearGradient(
                colors: [.accentColor, .accentColor.opacity(0.85)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: .black.opacity(0.15), radius: 8, y: 2)
        )
    }

    // MARK: - Metadata Row

    private var metadataRow: some View {
        HStack(spacing: 6) {
            Spacer(minLength: 0)
            Text(message.timestamp, style: .time)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary.opacity(0.65))
        }
        .frame(maxWidth: isImageMessage ? 480 : nil)
        .padding(.horizontal, 4)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var streamingIndicator: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
                .tint(.secondary)
            Text("Generating response...")
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Generated Image with Context Menu

private struct GeneratedImageView: View {
    let image: PlatformImage
    @StateObject private var imageSaver = ImageSaver()
    @State private var showSaveSuccess = false

    var body: some View {
        Image(uiImage: image)
            .resizable()
            .aspectRatio(image.size.width / image.size.height, contentMode: .fit)
            .frame(maxWidth: 480)
            .background(Color.black.opacity(0.02))
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.1), radius: 16, y: 4)
            .contextMenu {
                Button {
                    Task {
                        let result = await imageSaver.save(image)
                        if case .success = result { showSaveSuccess = true }
                    }
                } label: {
                    Label("Save to Photos", systemImage: "square.and.arrow.down")
                }
                ShareLink(
                    item: Image(uiImage: image),
                    preview: SharePreview("Generated Image", image: Image(uiImage: image))
                ) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
            } preview: {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
    }
}

// MARK: - Streaming Text with Typewriter Animation

/// Animates newly appended text character-by-character (50 chars/sec) with a blinking cursor.
struct StreamingText: View {
    let fullText: String

    @State private var displayedText = ""
    @State private var cursorVisible = true
    @State private var animationTask: Task<Void, Never>?

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            Text(displayedText)
                .textSelection(.enabled)

            Rectangle()
                .fill(Color.primary)
                .frame(width: 2, height: 18)
                .opacity(cursorVisible ? 1.0 : 0.0)
                .padding(.leading, 2)
        }
        .onChange(of: fullText) { _, newValue in animateAppendedText(newValue) }
        .onAppear {
            startCursorBlink()
            if !fullText.isEmpty { displayedText = fullText }
        }
        .onDisappear { animationTask?.cancel() }
    }

    private func animateAppendedText(_ newText: String) {
        animationTask?.cancel()

        // If text was reset or replaced, snap to the new value
        guard newText.count > displayedText.count, newText.hasPrefix(displayedText) else {
            displayedText = newText
            return
        }

        let start = displayedText.count
        animationTask = Task { @MainActor in
            for i in start..<newText.count {
                guard !Task.isCancelled else { break }
                let idx = newText.index(newText.startIndex, offsetBy: i + 1)
                displayedText = String(newText[..<idx])
                try? await Task.sleep(nanoseconds: 20_000_000) // 20 ms per character
            }
            if !Task.isCancelled { displayedText = newText }
        }
    }

    private func startCursorBlink() {
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            withAnimation(.linear(duration: 0.1)) { cursorVisible.toggle() }
        }
    }
}
