// 消息角色枚举
export enum MessageRole {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system'
}

// 消息类型枚举
export enum MessageType {
  TEXT = 'text',
  IMAGE = 'image',
  FILE = 'file',
  ERROR = 'error'
}

// 聊天消息接口
export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  messageType: MessageType
  timestamp: string
  metadata?: Record<string, any>
}

// 对话接口
export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
}

// 聊天请求接口
export interface ChatRequest {
  message: string
  conversationId?: string
  context?: ChatMessage[]
  temperature?: number
  maxTokens?: number
}

// 聊天响应接口
export interface ChatResponse {
  messageId: string
  content: string
  conversationId: string
  timestamp: string
  sources?: Source[]
  metadata?: Record<string, any>
}

// 来源接口
export interface Source {
  title: string
  url?: string
  score: number
}

// 对话接口
export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
  metadata?: Record<string, any>
}

// 聊天历史响应接口
export interface ChatHistoryResponse {
  conversations: Conversation[]
  total: number
  page: number
  size: number
}

// 搜索结果接口
export interface SearchResult {
  id: string
  title: string
  content: string
  url?: string
  score: number
  metadata?: Record<string, any>
}
