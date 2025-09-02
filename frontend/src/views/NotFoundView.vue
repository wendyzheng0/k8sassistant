<template>
  <div class="not-found-view">
    <el-result
      icon="error"
      title="404"
      sub-title="抱歉，您访问的页面不存在"
    >
      <template #extra>
        <el-button type="primary" @click="$router.push('/')">返回首页</el-button>
      </template>
    </el-result>
  </div>
</template>

<script setup lang="ts">
// 404 页面组件逻辑
</script>

<style scoped>
.not-found-view {
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 40px);
}
</style>
