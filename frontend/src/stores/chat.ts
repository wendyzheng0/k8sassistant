import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ChatMessage, ChatRequest, ChatResponse, Conversation } from '@/types/chat'
import { MessageRole, MessageType } from '@/types/chat'
import { chatAPI } from '@/api/chat'

export const useChatStore = defineStore('chat', () => {
  // State
  const conversations = ref<Conversation[]>([])
  const currentConversation = ref<Conversation | null>(null)
  const messages = ref<ChatMessage[]>([])
  const isLoading = ref(false)
  const isStreaming = ref(false)
  const error = ref<string | null>(null)

  // Export conversations
  const getConversations = computed(() => conversations.value)

  // Computed properties
  const hasMessages = computed(() => messages.value.length > 0)
  const currentConversationId = computed(() => currentConversation.value?.id)

  // Methods
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
      
      // 根据错误类型提供更详细的错误信息
      let errorContent = '抱歉，发送消息时出现错误，请稍后重试。'
      
      if (err instanceof Error) {
        if (err.message.includes('timeout') || err.message.includes('超时')) {
          errorContent = '请求超时，LLM响应时间较长，请稍后重试。'
        } else if (err.message.includes('Network Error') || err.message.includes('网络')) {
          errorContent = '网络连接错误，请检查网络连接后重试。'
        } else if (err.message.includes('500')) {
          errorContent = '服务器内部错误，请稍后重试。如果问题持续，请联系管理员。'
        } else if (err.message.includes('404')) {
          errorContent = '服务不可用，请检查后端服务是否正常运行。'
        }
      }
      
      // Add error message
      const errorMessage: ChatMessage = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: errorContent,
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
    
    // Add assistant reply
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

    // Update or create conversation
    updateConversation(response.conversationId, response.content)
  }

  const sendMessageStream = async (request: ChatRequest) => {
    isStreaming.value = true
    
    try {
      const stream = await chatAPI.sendMessageStream(request)
      const reader = stream.getReader()
      const decoder = new TextDecoder()

      // Create assistant message placeholder and get its index for reactive updates
      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: '...',
        messageType: MessageType.TEXT,
        timestamp: new Date().toISOString(),
        metadata: {}
      }
      messages.value.push(assistantMessage)
      const msgIndex = messages.value.length - 1

      let fullContent = ''
      let conversationId = ''
      let buffer = ''

      const handleRawSseEvent = (rawEvent: string) => {
        if (!rawEvent.trim()) return

        let eventType = 'message'
        const dataLines: string[] = []

        for (const line of rawEvent.split('\n')) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
            continue
          }
          if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trimStart())
          }
        }

        const dataStr = dataLines.join('\n')
        let payload: any = {}
        try {
          payload = dataStr ? JSON.parse(dataStr) : {}
        } catch (e) {
          console.warn('Failed to parse SSE payload:', dataStr, e)
        }

        if (eventType === 'meta') {
          // backend sends snake_case keys; keep both for safety
          conversationId = payload.conversation_id || payload.conversationId || conversationId
          const sources = payload.sources || []
          console.log('[SSE] Received meta event, sources:', sources.length)
          // Force Vue reactivity by replacing the object
          messages.value[msgIndex] = {
            ...messages.value[msgIndex],
            metadata: {
              ...(messages.value[msgIndex].metadata || {}),
              conversationId,
              sources
            }
          }
          return
        }

        if (eventType === 'delta') {
          const delta = payload.delta ?? ''
          fullContent += delta
          // Force Vue reactivity by replacing the object
          messages.value[msgIndex] = {
            ...messages.value[msgIndex],
            content: fullContent
          }
          return
        }

        if (eventType === 'done') {
          updateConversation(conversationId, fullContent)
          // Using an exception-free exit: we just mark a sentinel and let caller return
          throw new Error('__STREAM_DONE__')
        }

        if (eventType === 'error') {
          const errMsg = payload.error || '流式响应出错'
          throw new Error(errMsg)
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        // Normalize CRLF to LF so splitting is consistent across proxies/platforms
        buffer = buffer.replace(/\r\n/g, '\n')

        // Consume as many complete SSE events as possible.
        // Each event ends with a blank line: "\n\n"
        while (true) {
          const idx = buffer.indexOf('\n\n')
          if (idx === -1) break
          const rawEvent = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          try {
            handleRawSseEvent(rawEvent)
          } catch (e: any) {
            if (e?.message === '__STREAM_DONE__') return
            throw e
          }
        }
      }

      // Flush any remaining complete events when stream closes
      buffer = buffer.replace(/\r\n/g, '\n')
      while (true) {
        const idx = buffer.indexOf('\n\n')
        if (idx === -1) break
        const rawEvent = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        try {
          handleRawSseEvent(rawEvent)
        } catch (e: any) {
          if (e?.message === '__STREAM_DONE__') return
          throw e
        }
      }
    } finally {
      isStreaming.value = false
    }
  }

  const updateConversation = (conversationId: string, lastMessage: string) => {
    if (!currentConversation.value) {
      // Create new conversation
      currentConversation.value = {
        id: conversationId,
        title: lastMessage.slice(0, 50) + (lastMessage.length > 50 ? '...' : ''),
        messages: messages.value,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    } else {
      // Update existing conversation
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
    // State
    currentConversation,
    messages,
    isLoading,
    isStreaming,
    error,
    
    // Computed properties
    hasMessages,
    currentConversationId,
    conversations: getConversations,
    
    // Methods
    sendMessage,
    startNewConversation,
    loadConversation,
    clearMessages
  }
})
