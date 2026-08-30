import SwiftUI

/// Scrollable message history that auto-scrolls to the latest content.
///
/// Handles three scroll triggers:
/// 1. New message added → immediate scroll.
/// 2. Streaming text update → debounced scroll (100 ms) to avoid layout thrashing.
/// 3. Image generation started → scroll to the progress preview card.
struct ConversationView: View {
    let conversation: Conversation
    let isGenerating: Bool
    var generationModel: MobileOModel?

    @State private var scrollDebounceTimer: Timer?

    // MARK: - Body

    var body: some View {
        ScrollViewReader { proxy in
            LazyVStack(spacing: 16, pinnedViews: []) {
                messageList
                generatingPreview
            }
            .padding(.top, 20)
            .padding(.bottom, 180)
            .compositingGroup()
            .animation(.spring(response: 0.5, dampingFraction: 0.95), value: conversation.messages.count)
            .onChange(of: conversation.messages.count) { _, _ in
                scrollToLatest(proxy: proxy)
            }
            .onChange(of: conversation.messages.last?.content) { _, _ in
                debouncedScrollDuringStreaming(proxy: proxy)
            }
            .onChange(of: isGenerating) { wasGenerating, nowGenerating in
                scrollToGeneratingPreviewIfNeeded(proxy: proxy, wasGenerating: wasGenerating, nowGenerating: nowGenerating)
            }
            .onDisappear { scrollDebounceTimer?.invalidate() }
        }
    }

    // MARK: - Subviews

    private var messageList: some View {
        ForEach(conversation.messages) { message in
            let isLast = message.id == conversation.messages.last?.id
            let isStreamingThis = isGenerating && isLast && message.role == .assistant

            MessageBubbleView(
                message: message,
                isStreaming: isStreamingThis
            )
            .id(message.id)
            .layoutPriority(1)
            .transition(.asymmetric(
                insertion: .move(edge: message.role == .user ? .trailing : .leading)
                    .combined(with: .opacity)
                    .animation(.spring(response: 0.5, dampingFraction: 0.95)),
                removal: .scale(scale: 0.96)
                    .combined(with: .opacity)
                    .animation(.spring(response: 0.4, dampingFraction: 0.9))
            ))
        }
    }

    /// Progress card shown at the bottom while image generation is running.
    @ViewBuilder
    private var generatingPreview: some View {
        if let model = generationModel, model.running, isGenerating {
            HStack(alignment: .bottom, spacing: 12) {
                GeneratingImageView(
                    currentStep: model.currentStep,
                    totalSteps: model.totalSteps
                )
                .frame(maxWidth: 600, alignment: .leading)

                Spacer(minLength: 80)
            }
            .padding(.horizontal, 16)
            .id("generating-preview")
        }
    }

    // MARK: - Scroll Helpers

    private func scrollToLatest(proxy: ScrollViewProxy) {
        guard let lastMessage = conversation.messages.last else { return }
        withAnimation(.spring(response: 0.5, dampingFraction: 0.95)) {
            proxy.scrollTo(lastMessage.id, anchor: .top)
        }
    }

    /// Debounces scroll during text streaming to avoid excessive layout passes.
    private func debouncedScrollDuringStreaming(proxy: ScrollViewProxy) {
        guard generationModel?.running != true,
              let lastMessage = conversation.messages.last else { return }

        scrollDebounceTimer?.invalidate()
        scrollDebounceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: false) { _ in
            withAnimation(.spring(response: 0.5, dampingFraction: 0.95)) {
                proxy.scrollTo(lastMessage.id, anchor: .top)
            }
        }
    }

    private func scrollToGeneratingPreviewIfNeeded(proxy: ScrollViewProxy, wasGenerating: Bool, nowGenerating: Bool) {
        let isImageGen = generationModel?.running == true || wasGenerating
        guard nowGenerating, isImageGen else { return }
        withAnimation(.spring(response: 0.5, dampingFraction: 0.95)) {
            proxy.scrollTo("generating-preview", anchor: .top)
        }
    }
}
