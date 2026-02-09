<template>
  <div class="user-menu">
    <el-dropdown trigger="click" @command="handleCommand">
      <div class="user-menu-trigger">
        <el-avatar :icon="User" :size="32" />
        <span class="user-name">{{ userName }}</span>
        <el-icon class="dropdown-icon"><CaretBottom /></el-icon>
      </div>

      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="logout">
            <el-icon><SwitchButton /></el-icon>
            <span>登出</span>
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { User, SwitchButton, CaretBottom } from '@element-plus/icons-vue'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

const userName = computed(() => {
  if (!userStore.user) {
    return '匿名'
  }
  return userStore.user.username || userStore.user.email || '用户'
})

const handleCommand = (command: string) => {
  if (command === 'logout') {
    userStore.logout()
  }
}
</script>

<style scoped>
.user-menu {
  position: relative;
}

.user-menu-trigger {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.user-menu-trigger:hover {
  background-color: #f0f2f5;
}

.user-name {
  font-size: 14px;
  color: #606266;
  white-space: nowrap;
}

.dropdown-icon {
  font-size: 12px;
  color: #909399;
}
</style>