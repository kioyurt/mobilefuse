import Foundation
import SwiftUI

/// Observable conversation model holding the ordered message history and streaming state.
@Observable
class Conversation {
    var messages: [ChatMessage] = []
    var isGenerating: Bool = false
    var streamingMessageId: UUID?

    /// Append a new message and return its ID (for later metric updates).
    @discardableResult
    func addMessage(role: MessageRole, content: MessageContent, timeToFirstToken: TimeInterval? = nil,
                   totalTime: TimeInterval? = nil, tokensPerSecond: Double? = nil) -> UUID {
        let message = ChatMessage(role: role, content: content, timeToFirstToken: timeToFirstToken,
                                 totalTime: totalTime, tokensPerSecond: tokensPerSecond)
        messages.append(message)
        return message.id
    }

    /// Update the last message's text content during streaming
    func updateLastMessage(text: String) {
        guard !messages.isEmpty else { return }
        messages[messages.count - 1].content = .text(text)
    }

    /// Update performance metrics on an existing message by its ID.
    func updateMessageMetrics(id: UUID, timeToFirstToken: TimeInterval? = nil,
                             totalTime: TimeInterval? = nil, tokensPerSecond: Double? = nil) {
        if let index = messages.firstIndex(where: { $0.id == id }) {
            if let ttft = timeToFirstToken {
                messages[index].timeToFirstToken = ttft
            }
            if let total = totalTime {
                messages[index].totalTime = total
            }
            if let tokPerSec = tokensPerSecond {
                messages[index].tokensPerSecond = tokPerSec
            }
        }
    }

    func clear() {
        messages.removeAll()
        streamingMessageId = nil
    }

    /// Messages as dictionaries for model processing
    func getMessagesForModel() -> [[String: String]] {
        return messages.map { $0.toDict() }
    }

    var isEmpty: Bool { messages.isEmpty }
}
