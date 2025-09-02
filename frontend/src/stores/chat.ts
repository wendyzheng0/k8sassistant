import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ChatMessage, ChatRequest, ChatResponse, Conversation } from '@/types/chat'
import { MessageRole, MessageType } from '@/types/chat'
import { chatAPI } from '@/api/chat'

export const useChatStore = defineStore('chat', () => {
  // 状态
  const conversations = ref<Conversation[]>([])
  const currentConversation = ref<Conversation | null>(null)
  const messages = ref<ChatMessage[]>([])
  const isLoading = ref(false)
  const isStreaming = ref(false)
  const error = ref<string | null>(null)

  // 导出 conversations
  const getConversations = computed(() => conversations.value)

  // 计算属性
  const hasMessages = computed(() => messages.value.length > 0)
  const currentConversationId = computed(() => currentConversation.value?.id)

  // 方法
  const sendMessage = async (content: string, useStream = false) => {
    if (!content.trim()) return

    try {
      isLoading.value = true
      error.value = null

      // 添加用户消息
      const userMessage: ChatMessage = {
        id: generateId(),
        role: MessageRole.USER,
        content: content.trim(),
        messageType: MessageType.TEXT,
        timestamp: new Date().toISOString()
      }
      messages.value.push(userMessage)

      // 准备请求
      const request: ChatRequest = {
        message: content.trim(),
        conversationId: currentConversationId.value,
        context: messages.value.slice(-10), // 最近10条消息作为上下文
        temperature: 0.7,
        maxTokens: 2048
      }

      if (useStream) {
        await sendMessageStream(request)
      } else {
        await sendMessageSync(request)
      }

    } catch (err) {
      console.error('发送消息失败:', err)
      error.value = err instanceof Error ? err.message : '发送消息失败'
      
      // 添加错误消息
      const errorMessage: ChatMessage = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: '抱歉，发送消息时出现错误，请稍后重试。',
        messageType: MessageType.ERROR,
        timestamp: new Date().toISOString()
      }
      messages.value.push(errorMessage)
    } finally {
      isLoading.value = false
    }
  }

  const sendMessageSync = async (request: ChatRequest) => {
    const response: ChatResponse = await chatAPI.sendMessage(request)
    
    // 添加助手回复
    const assistantMessage: ChatMessage = {
      id: response.messageId,
      role: MessageRole.ASSISTANT,
      content: response.content,
      messageType: MessageType.TEXT,
      timestamp: response.timestamp,
      metadata: {
        sources: response.sources,
        conversationId: response.conversationId
      }
    }
    messages.value.push(assistantMessage)

    // 更新或创建对话
    updateConversation(response.conversationId, response.content)
  }

  const sendMessageStream = async (request: ChatRequest) => {
    isStreaming.value = true
    
    try {
      const stream = await chatAPI.sendMessageStream(request)
      const reader = stream.getReader()
      const decoder = new TextDecoder()

      // 创建助手消息占位符
      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: '',
        messageType: MessageType.TEXT,
        timestamp: new Date().toISOString()
      }
      messages.value.push(assistantMessage)

      let fullContent = ''
      let conversationId = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            
            if (data === '[DONE]') {
              // 流结束
              updateConversation(conversationId, fullContent)
              return
            }

            // 累积内容
            fullContent += data
            assistantMessage.content = fullContent
          }
        }
      }
    } finally {
      isStreaming.value = false
    }
  }

  const updateConversation = (conversationId: string, lastMessage: string) => {
    if (!currentConversation.value) {
      // 创建新对话
      currentConversation.value = {
        id: conversationId,
        title: lastMessage.slice(0, 50) + (lastMessage.length > 50 ? '...' : ''),
        messages: messages.value,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    } else {
      // 更新现有对话
      currentConversation.value.updatedAt = new Date().toISOString()
      currentConversation.value.messages = messages.value
    }
  }

  const startNewConversation = () => {
    currentConversation.value = null
    messages.value = []
    error.value = null
  }

  const loadConversation = (conversation: Conversation) => {
    currentConversation.value = conversation
    messages.value = [...conversation.messages]
    error.value = null
  }

  const clearMessages = () => {
    messages.value = []
    error.value = null
  }

  const generateId = () => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2)
  }

  return {
    // 状态
    currentConversation,
    messages,
    isLoading,
    isStreaming,
    error,
    
    // 计算属性
    hasMessages,
    currentConversationId,
    conversations: getConversations,
    
    // 方法
    sendMessage,
    startNewConversation,
    loadConversation,
    clearMessages
  }
})
