import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authAPI, getAuthToken, setAuthToken, clearAuthToken } from '@/api/chat'

export interface User {
  id: string
  username: string
  email: string
  is_anonymous: boolean
  created_at: string
}

export const useUserStore = defineStore('user', () => {
  // 状态
  const user = ref<User | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // 计算属性
  const isAuthenticated = computed(() => !!user.value && !user.value.is_anonymous)
  const isAnonymous = computed(() => !!user.value && user.value.is_anonymous)
  const userId = computed(() => user.value?.id || null)
  const userName = computed(() => {
    if (!user.value) return '匿名'
    return user.value.username || user.value.email || '用户'
  })

  // 初始化 - 检查是否已登录
  async function init() {
    const token = getAuthToken()
    if (token) {
      // 有 token，尝试获取用户信息
      try {
        const userData: any = await authAPI.getCurrentUser()
        user.value = userData as User | null
      } catch (e) {
        // Token 无效，清除
        console.error('Failed to get current user:', e)
        clearAuthToken()
        user.value = null
      }
    }
  }

  // 用户注册
  async function register(username: string, email: string, password: string) {
    isLoading.value = true
    error.value = null

    try {
      const userData: any = await authAPI.register({ username, email, password })
      user.value = userData as User
      error.value = null
    } catch (e: any) {
      console.error('Registration failed:', e)
      error.value = e.response?.data?.detail || '注册失败，请稍后重试'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  // 用户登录
  async function login(email: string, password: string) {
    isLoading.value = true
    error.value = null

    try {
      const response: any = await authAPI.login(email, password)
      // authAPI.login 会自动保存 token 并触发合并
      // 获取用户信息
      const userData: any = await authAPI.getCurrentUser()
      user.value = userData as User | null
      error.value = null
    } catch (e: any) {
      console.error('Login failed:', e)
      error.value = e.response?.data?.detail || '登录失败，请检查用户名和密码'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  // 用户登出
  async function logout() {
    try {
      await authAPI.logout()
      user.value = null
      error.value = null
    } catch (e: any) {
      console.error('Logout failed:', e)
      // 登出失败也清除本地状态
      user.value = null
      setAuthToken('')
      clearAuthToken()
    }
  }

  // 更新用户资料
  async function updateUser(data: {
    username?: string
    email?: string
    password?: string
  }) {
    if (!user.value) return

    isLoading.value = true
    error.value = null

    try {
      const updatedUser: any = await authAPI.updateUser(user.value.id!, data)
      user.value = { ...user.value!, ...updatedUser }
      error.value = null
    } catch (e: any) {
      console.error('Update user failed:', e)
      error.value = e.response?.data?.detail || '更新失败，请稍后重试'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  // 删除用户账户
  async function deleteUser() {
    if (!user.value) return

    isLoading.value = true
    error.value = null

    try {
      await authAPI.deleteUser(user.value.id!)
      user.value = null
      error.value = null
    } catch (e: any) {
      console.error('Delete user failed:', e)
      error.value = e.response?.data?.detail || '删除失败，请稍后重试'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  // 清除错误
  function clearError() {
    error.value = null
  }

  return {
    // 状态
    user,
    isLoading,
    error,
    // 计算属性
    isAuthenticated,
    isAnonymous,
    userId,
    // 方法
    init,
    register,
    login,
    logout,
    updateUser,
    deleteUser,
    clearError
  }
})
