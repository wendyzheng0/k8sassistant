<template>
  <el-dialog
    v-model="dialogVisible"
    title="用户登录"
    width="400px"
    :before-close="handleClose"
    :destroy-on-close="true"
    :close-on-click-modal="false"
  >
    <div class="login-tabs">
      <el-tabs v-model="activeTab" class="demo-tabs">
        <el-tab-pane label="登录" name="login">
          <el-form
            ref="loginFormRef"
            :model="loginForm"
            :rules="loginRules"
            label-width="80px"
          >
            <el-form-item label="用户名" prop="username">
              <el-input
                v-model="loginForm.username"
                placeholder="请输入用户名"
                type="email"
              />
            </el-form-item>
            <el-form-item label="密码" prop="password">
              <el-input
                v-model="loginForm.password"
                placeholder="请输入密码"
                type="password"
                show-password
              />
            </el-form-item>
            <el-form-item>
              <el-button
                type="primary"
                @click="handleLogin"
                :loading="isLoading"
                style="width: 100%"
              >
                登录
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="注册" name="register">
          <el-form
            ref="registerFormRef"
            :model="registerForm"
            :rules="registerRules"
            label-width="80px"
          >
            <el-form-item label="用户名" prop="username">
              <el-input
                v-model="registerForm.username"
                placeholder="请输入用户名"
              />
            </el-form-item>
            <el-form-item label="邮箱" prop="email">
              <el-input
                v-model="registerForm.email"
                placeholder="请输入邮箱"
                type="email"
              />
            </el-form-item>
            <el-form-item label="密码" prop="password">
              <el-input
                v-model="registerForm.password"
                placeholder="请输入密码"
                type="password"
                show-password
              />
            </el-form-item>
            <el-form-item>
              <el-button
                type="primary"
                @click="handleRegister"
                :loading="isLoading"
                style="width: 100%"
              >
                注册
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage, ElForm } from 'element-plus'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

// Dialog visibility
const dialogVisible = ref(false)
const activeTab = ref('login')

// Loading state
const isLoading = ref(false)

// Form refs
const loginFormRef = ref<InstanceType<typeof ElForm>>()
const registerFormRef = ref<InstanceType<typeof ElForm>>()

// Login form
const loginForm = reactive({
  username: '',
  password: ''
})

// Register form
const registerForm = reactive({
  username: '',
  email: '',
  password: ''
})

// Validation rules
const loginRules = reactive({
  email: [
    { required: true, message: '请输入邮箱地址', trigger: 'blur' },
    { type: 'email' as const, message: '请输入正确的邮箱地址', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
  ]
})

const registerRules = reactive({
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 20, message: '用户名长度在 3 到 20 个字符', trigger: 'blur' }
  ],
  email: [
    { required: true, message: '请输入邮箱地址', trigger: 'blur' },
    { type: 'email' as const, message: '请输入正确的邮箱地址', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
  ]
})

// Methods
const show = () => {
  dialogVisible.value = true
}

const handleClose = () => {
  dialogVisible.value = false
}

const handleLogin = async () => {
  if (!loginFormRef.value) return

  await loginFormRef.value.validate(async (valid) => {
    if (valid) {
      isLoading.value = true
      try {
        await userStore.login(loginForm.username, loginForm.password)
        ElMessage.success('登录成功')
        isLoading.value = false
        dialogVisible.value = false
      } catch (error: any) {
        isLoading.value = false
        ElMessage.error(error.message || '登录失败')
      }
    }
  })
}

const handleRegister = async () => {
  if (!registerFormRef.value) return

  await registerFormRef.value.validate(async (valid) => {
    if (valid) {
      isLoading.value = true
      try {
        await userStore.register(registerForm.username, registerForm.email, registerForm.password)
        ElMessage.success('注册成功，请登录')
        activeTab.value = 'login'
        // Clear register form
        registerForm.username = ''
        registerForm.email = ''
        registerForm.password = ''
      } catch (error: any) {
        ElMessage.error(error.message || '注册失败')
      } finally {
        isLoading.value = false
      }
    }
  })
}

defineExpose({
  show
})
</script>

<style scoped>
.demo-tabs >>> .el-tabs__item {
  font-size: 14px;
  color: #606266;
}

.demo-tabs >>> .el-tabs__item.is-active {
  color: #409eff;
  font-weight: bold;
}

.login-tabs {
  padding: 0 20px;
}

/* 修复：确保 dialog 关闭后遮罩层被正确移除 */
:deep(.el-overlay) {
  z-index: 2000;
}

/* 修复：防止 loading 状态下按钮样式影响 dialog 关闭 */
:deep(.el-button.is-loading) {
  pointer-events: none;
}
</style>