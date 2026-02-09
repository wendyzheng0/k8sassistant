import axios from 'axios'
import type { ChatRequest, ChatResponse, ChatHistoryResponse } from '@/types/chat'

// 匿名用户 ID 管理
const anonymous_user_id_key = 'anonymous_user_id'

// 获取或创建匿名用户 ID
export function getOrCreateAnonymousUserId(): string {
  let anonymousId = localStorage.getItem(anonymous_user_id_key)
  if (!anonymousId) {
    // 生成新的匿名用户 ID
    anonymousId = crypto.randomUUID()
    localStorage.setItem(anonymous_user_id_key, anonymousId)
  }
  return anonymousId
}

// 获取认证token
export function getAuthToken(): string | null {
  return localStorage.getItem('auth_token')
}

// 设置认证token
export function setAuthToken(token: string): void {
  localStorage.setItem('auth_token', token)
}

// 清除认证token
export function clearAuthToken(): void {
  localStorage.removeItem('auth_token')
}

// 创建 axios 实例
const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120000, // 增加到2分钟，给LLM更多时间
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 添加认证 token
    const token = getAuthToken()
    if (token) {
      config.headers = config.headers || {}
      config.headers['Authorization'] = `Bearer ${token}`
    } else {
      // 添加匿名用户 ID
      const anonymousId = getOrCreateAnonymousUserId()
      config.headers = config.headers || {}
      config.headers['X-Anonymous-User-ID'] = anonymousId
    }
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
    // 处理 401 未授权错误
    if (error.response && error.response.status === 401) {
      clearAuthToken()
    }
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
      // Add auth headers
      const token = getAuthToken()
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      }
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      } else {
        const anonymousId = getOrCreateAnonymousUserId()
        headers['X-Anonymous-User-ID'] = anonymousId
      }

      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: headers,
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

// 用户认证API
export const authAPI = {
  // 用户注册
  async register(data: {
    username: string
    email: string
    password: string
  }): Promise<{ id: string, username: string, email: string }> {
    return api.post('/users/register', data)
  },

  // 用户登录
  async login(username: string, password: string): Promise<any> {
    const response = await api.post('/users/login', { username, password })
    // 保存 token
    if ((response as any).access_token) {
      setAuthToken((response as any).access_token)
      // 登录后合并匿名用户数据
      const anonymousId = getOrCreateAnonymousUserId()
      await this.mergeAnonymousUser(anonymousId)
    }
    return response
  },

  // 获取当前用户信息
  async getCurrentUser(): Promise<any> {
    return api.get('/users/me')
  },

  // 获取用户信息
  async getUser(userId: string): Promise<any> {
    return api.get(`/users/${userId}`)
  },

  // 更新用户资料
  async updateUser(userId: string, data: {
    username?: string
    email?: string
    password?: string
  }): Promise<any> {
    return api.put(`/users/${userId}`, data)
  },

  // 删除用户账户
  async deleteUser(userId: string): Promise<any> {
    const response = await api.delete(`/users/${userId}`)
    clearAuthToken()
    return response
  },

  // 合并匿名用户数据到登录用户
  async mergeAnonymousUser(anonymousUserId: string): Promise<any> {
    try {
      return await api.post('/users/merge', { anonymous_user_id: anonymousUserId })
    } catch (error) {
      console.error('Merge failed:', error)
      // 合并失败不影响登录，静默失败
      return { message: 'Merge skipped' } as any
    }
  },

  // 用户登出
  async logout(): Promise<void> {
    clearAuthToken()
    // 不删除匿名用户 ID，保留本地历史
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
