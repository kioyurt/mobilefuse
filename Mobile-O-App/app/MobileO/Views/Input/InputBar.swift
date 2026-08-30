import SwiftUI
import PhotosUI
import UIKit

/// Floating bottom input bar with image attachment, text field, mode indicators, and send/cancel button.
///
/// Detects prompt keywords to switch modes:
/// - `"generate …"` → purple "Image Generation" pill
///
/// Also shows scrollable example chips when in generate mode with an active conversation.
struct InputBar: View {
    @Binding var prompt: String
    @Binding var selectedImage: PlatformImage?
    @Binding var photoPickerItem: PhotosPickerItem?

    let conversation: Conversation
    let isGenerating: Bool
    let onSend: () -> Void
    let onCancel: () -> Void
    let onClearConversation: () -> Void
    let onReset: () -> Void

    @FocusState private var isTextFieldFocused: Bool

    // MARK: - Computed Properties

    private var isGenerateMode: Bool {
        let lowercased = prompt.lowercased()
        return lowercased == "generate" || lowercased.hasPrefix("generate ")
    }

    private var placeholderText: String {
        if selectedImage != nil { return "Ask about this image..." }
        if isGenerateMode { return "Describe what to create..." }
        return "Message Mobile-O"
    }

    private var dynamicHeight: CGFloat {
        let minHeight: CGFloat = 40
        let maxHeight: CGFloat = 150
        guard !prompt.isEmpty else { return minHeight }

        let font = UIFont.systemFont(ofSize: 15.5, weight: .regular)
        let rect = (prompt as NSString).boundingRect(
            with: CGSize(width: 400, height: CGFloat.greatestFiniteMagnitude),
            options: .usesLineFragmentOrigin,
            attributes: [.font: font],
            context: nil
        )
        return max(minHeight, min(rect.height + 20, maxHeight))
    }

    private var sendButtonColor: Color {
        if isGenerating { return .red }
        if isGenerateMode { return .purple }
        return .accentColor
    }

    private var canSend: Bool {
        isGenerating || !prompt.isEmpty || selectedImage != nil
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            imagePreview
            VStack(spacing: 10) {
                exampleChips
                textField
                actionRow
            }
            .padding(.horizontal, 22)
            .padding(.top, 8)
        }
        .safeAreaPadding(.bottom)
        .background { barBackground }
        .animation(.spring(response: 0.35, dampingFraction: 0.75), value: selectedImage != nil)
        .animation(.spring(response: 0.35, dampingFraction: 0.75), value: isGenerateMode)
    }

    // MARK: - Subviews

    @ViewBuilder
    private var imagePreview: some View {
        if let selectedImage {
            ImagePreviewCard(image: selectedImage, onDismiss: onReset)
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .transition(.scale(scale: 0.96).combined(with: .opacity))
        }
    }

    @ViewBuilder
    private var exampleChips: some View {
        if isGenerateMode {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    GenerateExampleChip(text: "Mountain lake", onTap: { prompt = "Generate a serene mountain lake surrounded by autumn trees" })
                    GenerateExampleChip(text: "Cozy cabin", onTap: { prompt = "Generate a cozy wooden cabin in a snowy forest with warm lights" })
                    GenerateExampleChip(text: "Ocean sunset", onTap: { prompt = "Generate a peaceful ocean sunset with vibrant orange and pink clouds" })
                    GenerateExampleChip(text: "City skyline", onTap: { prompt = "Generate a modern city skyline at night with illuminated buildings" })
                    GenerateExampleChip(text: "Red rose", onTap: { prompt = "Generate a red rose with morning dew drops" })
                    GenerateExampleChip(text: "Forest path", onTap: { prompt = "Generate a magical forest path with sunlight filtering through trees" })
                }
                .padding(.leading, 22)
                .padding(.trailing, 8)
                .padding(.top, 8)
            }
            .padding(.horizontal, -22)
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    private var textField: some View {
        ZStack(alignment: .topLeading) {
            if prompt.isEmpty {
                Text(placeholderText)
                    .font(.system(size: 15.5, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
                    .padding(.top, 11)
                    .padding(.leading, 5)
                    .allowsHitTesting(false)
            }

            TextEditor(text: Binding(
                get: { prompt },
                set: { newValue in
                    if isGenerateMode && !newValue.lowercased().hasPrefix("generate") {
                        // User tried to delete the prefix — restore it
                        prompt = "Generate "
                    } else {
                        prompt = newValue
                    }
                }
            ))
                .font(.system(size: 15.5, weight: .regular, design: .rounded))
                .scrollContentBackground(.hidden)
                .background(Color.clear)
                .frame(height: dynamicHeight)
                .focused($isTextFieldFocused)
                .scrollDisabled(dynamicHeight < 150)
        }
        .animation(.easeInOut(duration: 0.2), value: dynamicHeight)
    }

    private var actionRow: some View {
        HStack(spacing: 4) {
            ImagePickerButton(
                selectedImage: $selectedImage,
                photoPickerItem: $photoPickerItem,
                onReset: onReset
            )
            pasteButton
            createModeToggle
            Spacer()
            sendButton
        }
    }

    private var createModeToggle: some View {
        Button {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            if isGenerateMode {
                let lowercased = prompt.lowercased()
                if lowercased == "generate" {
                    prompt = ""
                } else if lowercased.hasPrefix("generate ") {
                    prompt = String(prompt.dropFirst("generate ".count))
                }
            } else {
                prompt = "Generate " + prompt
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "sparkles")
                    .font(.system(size: 14, weight: .semibold))
                Text("Image Gen")
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
            }
            .foregroundStyle(isGenerateMode ? .white : .purple)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                Capsule()
                    .fill(isGenerateMode ? Color.purple : Color.purple.opacity(0.12))
            )
        }
        .buttonStyle(.plain)
    }

    private var pasteButton: some View {
        Button {
            if let clipboardText = UIPasteboard.general.string {
                prompt += clipboardText
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
            }
        } label: {
            Image(systemName: "doc.on.clipboard")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 38, height: 38)
        }
        .buttonStyle(.plain)
        .opacity(UIPasteboard.general.hasStrings ? 1.0 : 0.3)
        .disabled(!UIPasteboard.general.hasStrings)
    }

    private var sendButton: some View {
        Button {
            if isGenerating {
                onCancel()
            } else {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                onSend()
            }
        } label: {
            Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 38, height: 38)
                .background(
                    Circle()
                        .fill(sendButtonColor)
                        .shadow(color: sendButtonColor.opacity(0.3), radius: 6, y: 2)
                )
        }
        .buttonStyle(.plain)
        .disabled(!canSend)
        .opacity(canSend ? 1.0 : 0.4)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isGenerating)
    }

    private var barBackground: some View {
        GeometryReader { geometry in
            ZStack {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(.ultraThinMaterial)
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(Color.black.opacity(0.15))
                    .blendMode(.multiply)
            }
            .frame(height: geometry.size.height + geometry.safeAreaInsets.bottom)
            .shadow(color: .black.opacity(0.2), radius: 15, y: -3)
            .ignoresSafeArea(edges: .bottom)
        }
    }
}

// MARK: - Generate Example Chip

private struct GenerateExampleChip: View {
    let text: String
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.purple)
                Text(text)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.primary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Capsule().fill(.purple.opacity(0.08)))
            .overlay(Capsule().strokeBorder(.purple.opacity(0.2), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}
