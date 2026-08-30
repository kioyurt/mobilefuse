import Foundation
import SwiftUI

/// Message content: text, image, or text with an attached image
enum MessageContent: Equatable {
    case text(String)
    case image(PlatformImage)
    case textWithImage(String, PlatformImage)

    static func == (lhs: MessageContent, rhs: MessageContent) -> Bool {
        switch (lhs, rhs) {
        case (.text(let lText), .text(let rText)):
            return lText == rText
        case (.image(let lImage), .image(let rImage)):
            return lImage.isEqual(rImage)
        case (.textWithImage(let lText, let lImage), .textWithImage(let rText, let rImage)):
            return lText == rText && lImage.isEqual(rImage)
        default:
            return false
        }
    }
}

/// Sender role for chat messages.
enum MessageRole: String, Codable {
    case user = "user"
    case assistant = "assistant"
    case system = "system"
}

/// A single message in a chat conversation with optional performance metrics
struct ChatMessage: Identifiable {
    let id: UUID
    let role: MessageRole
    var content: MessageContent
    let timestamp: Date
    var timeToFirstToken: TimeInterval?
    var totalTime: TimeInterval?
    var tokensPerSecond: Double?

    static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        return lhs.id == rhs.id &&
               lhs.role == rhs.role &&
               lhs.content == rhs.content &&
               lhs.timestamp == rhs.timestamp
    }

    init(id: UUID = UUID(), role: MessageRole, content: MessageContent, timestamp: Date = Date(),
         timeToFirstToken: TimeInterval? = nil, totalTime: TimeInterval? = nil, tokensPerSecond: Double? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.timeToFirstToken = timeToFirstToken
        self.totalTime = totalTime
        self.tokensPerSecond = tokensPerSecond
    }

    /// Text portion of the content, or empty string for image-only messages
    var textContent: String {
        switch content {
        case .text(let text):
            return text
        case .image:
            return ""
        case .textWithImage(let text, _):
            return text
        }
    }

    /// Image portion of the content, if present
    var imageContent: PlatformImage? {
        switch content {
        case .text:
            return nil
        case .image(let image):
            return image
        case .textWithImage(_, let image):
            return image
        }
    }

    /// Convert to dictionary format for model processing
    func toDict() -> [String: String] {
        return ["role": role.rawValue, "content": textContent]
    }
}
