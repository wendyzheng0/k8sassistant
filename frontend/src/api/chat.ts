import axios from 'axios'
import type { ChatRequest, ChatResponse, ChatHistoryResponse } from '@/types/chat'

// 创建 axios 实例
const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 增加到2分钟，给LLM更多时间
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
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 120000) // 2分钟超时
    
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(request),
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return response.body!
    } catch (error: any) {
      clearTimeout(timeoutId)
      if (error.name === 'AbortError') {
        throw new Error('请求超时，请稍后重试')
      }
      throw error
    }
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

// 管理界面API类型
export interface AdminTreeNode {
  label: string
  path: string
  is_file: boolean
  children?: AdminTreeNode[]
}

export interface AdminChunk {
  id: string
  content: string
  doc_id?: string
  file_path?: string
  chunk_index?: number
  metadata?: Record<string, any>
}

export interface AdminStats {
  collection_name?: string
  total_chunks?: number
  vector_dimension?: number
  status?: string
}

// 管理界面API
export const adminAPI = {
  // 获取文档树结构
  async getDocumentTree(): Promise<{ tree: AdminTreeNode[], base_path: string }> {
    return api.get('/admin/document-tree')
  },

  // 根据文件路径获取分块
  async getChunksByPath(filePath: string): Promise<{ file_path: string, chunks: AdminChunk[], total: number }> {
    return api.get('/admin/chunks', {
      params: { file_path: filePath }
    })
  },

  // 获取管理统计信息
  async getStats(): Promise<AdminStats> {
    return api.get('/admin/stats')
  }
}

export default api
