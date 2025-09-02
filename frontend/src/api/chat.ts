import axios from 'axios'
import type { ChatRequest, ChatResponse, ChatHistoryResponse } from '@/types/chat'

// 创建 axios 实例
const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// 聊天API
export const chatAPI = {
  // 发送聊天消息
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    return api.post('/chat', request)
  },

  // 流式聊天
  async sendMessageStream(request: ChatRequest): Promise<ReadableStream> {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.body!
  },

  // 获取聊天历史
  async getChatHistory(page = 1, size = 20): Promise<ChatHistoryResponse> {
    return api.get('/chat/history', {
      params: { page, size }
    })
  },

  // 删除对话
  async deleteConversation(conversationId: string): Promise<void> {
    return api.delete(`/chat/history/${conversationId}`)
  }
}

// 文档搜索API
export const documentAPI = {
  // 搜索文档
  async searchDocuments(query: string, topK = 5) {
    return api.get('/documents/search', {
      params: { query, top_k: topK }
    })
  },

  // 上传文档
  async uploadDocument(file: File) {
    const formData = new FormData()
    formData.append('file', file)
    
    return api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  // 获取文档统计
  async getDocumentStats() {
    return api.get('/documents/stats')
  },

  // 删除文档
  async deleteDocument(documentId: string) {
    return api.delete(`/documents/${documentId}`)
  }
}

// 健康检查API
export const healthAPI = {
  // 基础健康检查
  async healthCheck() {
    return api.get('/health')
  },

  // 详细健康检查
  async detailedHealthCheck() {
    return api.get('/health/detailed')
  },

  // 就绪检查
  async readinessCheck() {
    return api.get('/health/ready')
  }
}

export default api
