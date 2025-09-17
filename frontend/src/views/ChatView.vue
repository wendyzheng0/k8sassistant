<template>
  <div class="chat-container">
    <!-- Sidebar -->
    <div class="sidebar">
      <div class="sidebar-header">
        <h2>K8s Assistant</h2>
        <el-button 
          type="primary" 
          @click="startNewChat"
          :icon="Plus"
          size="small"
        >
          新对话
        </el-button>
      </div>
      
      <div class="sidebar-content">
        <el-menu
          :default-active="currentConversationId"
          class="conversation-list"
        >
          <el-menu-item 
            v-for="conversation in conversations" 
            :key="conversation.id"
            :index="conversation.id"
            @click="loadConversation(conversation)"
          >
            <el-icon><ChatDotRound /></el-icon>
            <span>{{ conversation.title }}</span>
          </el-menu-item>
        </el-menu>
      </div>
    </div>

    <!-- Main chat area -->
    <div class="main-content">
      <!-- Chat messages area -->
      <div class="chat-messages" ref="messagesContainer">
        <div v-if="!hasMessages" class="empty-state">
          <el-icon size="64" color="#909399"><ChatDotRound /></el-icon>
          <h3>欢迎使用 K8s 智能助手</h3>
          <p>我可以帮助您解答 Kubernetes 相关的问题</p>
          <div class="example-questions">
            <el-button 
              v-for="question in exampleQuestions" 
              :key="question"
              @click="sendExampleQuestion(question)"
              size="small"
              type="info"
              plain
            >
              {{ question }}
            </el-button>
          </div>
        </div>

        <div v-else class="messages-list">
          <div 
            v-for="message in messages" 
            :key="message.id"
            :class="['message', `message-${message.role}`]"
          >
            <div class="message-avatar">
              <el-avatar 
                :icon="message.role === 'user' ? User : Service"
                :size="32"
              />
            </div>
            <div class="message-content">
              <div class="message-header">
                <span class="message-role">
                  {{ message.role === 'user' ? '您' : 'K8s Assistant' }}
                </span>
                <span class="message-time">
                  {{ formatTime(message.timestamp) }}
                </span>
              </div>
              <div class="message-text" v-html="formatMessage(message.content)"></div>
              
              <!-- Show sources -->
              <div v-if="message.metadata?.sources?.length" class="message-sources">
                <div class="sources-header">
                  <span class="sources-title">参考来源</span>
                </div>
                <div 
                  v-for="(source, index) in message.metadata.sources" 
                  :key="source.title"
                  class="source-item"
                >
                  <div 
                    class="source-header"
                    @click="toggleSourceExpansion(message.id, index)"
                  >
                    <div class="source-title-section">
                      <el-icon class="expand-icon" :class="{ 'expanded': isSourceExpanded(message.id, index) }">
                        <ArrowRight />
                      </el-icon>
                      <el-link 
                        v-if="source.url" 
                        :href="source.url" 
                        target="_blank"
                        type="primary"
                        @click.stop
                      >
                        {{ source.title }}
                      </el-link>
                      <span v-else class="source-title-text">{{ source.title }}</span>
                    </div>
                    <el-tag size="small" type="info">
                      相关度: {{ (source.score * 100).toFixed(1) }}%
                    </el-tag>
                  </div>
                  <div 
                    v-if="isSourceExpanded(message.id, index)" 
                    class="source-content"
                  >
                    <div class="source-content-text">{{ source.content }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Loading state -->
          <div v-if="isLoading || isStreaming" class="message message-assistant">
            <div class="message-avatar">
              <el-avatar :icon="Service" :size="32" />
            </div>
            <div class="message-content">
              <div class="message-header">
                <span class="message-role">K8s Assistant</span>
              </div>
              <div class="loading-indicator">
                <el-icon class="is-loading"><Loading /></el-icon>
                <span>正在思考中...</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Input area -->
      <div class="chat-input">
        <div class="input-container">
          <el-input
            v-model="inputMessage"
            type="textarea"
            :rows="3"
            placeholder="请输入您的问题..."
            :disabled="isLoading || isStreaming"
            @keydown.enter.prevent="handleSend"
            resize="none"
          />
          <div class="input-actions">
            <el-button
              type="primary"
              :loading="isLoading || isStreaming"
              :disabled="!inputMessage.trim()"
              @click="handleSend"
              :icon="Promotion"
            >
              发送
            </el-button>
            <el-button
              @click="handleStreamSend"
              :loading="isStreaming"
              :disabled="!inputMessage.trim() || isLoading"
              :icon="VideoPlay"
            >
              流式发送
            </el-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { 
  Plus, 
  ChatDotRound, 
  User, 
  Service, 
  Loading, 
  Promotion, 
  VideoPlay,
  ArrowRight
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { marked } from 'marked'
import { useChatStore } from '@/stores/chat'
import type { Conversation } from '@/types/chat'

// 状态
const chatStore = useChatStore()
const inputMessage = ref('')
const messagesContainer = ref<HTMLElement>()

// 展开状态管理
const expandedSources = ref<Record<string, Set<number>>>({})

// Example questions
const exampleQuestions = [
  '如何创建一个 Kubernetes Pod？',
  '什么是 Deployment？',
  '如何配置 Service？',
  'Pod 和 Container 的区别是什么？',
  '如何查看集群状态？'
]

// Computed properties
const messages = computed(() => chatStore.messages)
const hasMessages = computed(() => chatStore.hasMessages)
const isLoading = computed(() => chatStore.isLoading)
const isStreaming = computed(() => chatStore.isStreaming)
const currentConversationId = computed(() => chatStore.currentConversationId)
const conversations = computed(() => chatStore.conversations)

// Methods
const handleSend = async () => {
  if (!inputMessage.value.trim() || isLoading.value) return
  
  const message = inputMessage.value
  inputMessage.value = ''
  
  await chatStore.sendMessage(message, false)
  scrollToBottom()
}

const handleStreamSend = async () => {
  if (!inputMessage.value.trim() || isStreaming.value) return
  
  const message = inputMessage.value
  inputMessage.value = ''
  
  await chatStore.sendMessage(message, true)
  scrollToBottom()
}

const sendExampleQuestion = async (question: string) => {
  inputMessage.value = question
  await handleSend()
}

const startNewChat = () => {
  chatStore.startNewConversation()
}

const loadConversation = (conversation: Conversation) => {
  chatStore.loadConversation(conversation)
}

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const formatTime = (timestamp: string) => {
  return new Date(timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

const formatMessage = (content: string) => {
  try {
    return marked(content)
  } catch {
    return content
  }
}

// 展开/收起source的方法
const toggleSourceExpansion = (messageId: string, sourceIndex: number) => {
  if (!expandedSources.value[messageId]) {
    expandedSources.value[messageId] = new Set()
  }
  
  const expandedSet = expandedSources.value[messageId]
  if (expandedSet.has(sourceIndex)) {
    expandedSet.delete(sourceIndex)
  } else {
    expandedSet.add(sourceIndex)
  }
}

const isSourceExpanded = (messageId: string, sourceIndex: number) => {
  return expandedSources.value[messageId]?.has(sourceIndex) || false
}

// Watch message changes, auto scroll to bottom
watch(messages, () => {
  scrollToBottom()
}, { deep: true })

// Scroll to bottom after component mount
onMounted(() => {
  scrollToBottom()
})
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  background-color: #f5f5f5;
}

.sidebar {
  width: 280px;
  background-color: #fff;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.sidebar-header h2 {
  margin: 0;
  color: #303133;
  font-size: 18px;
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
}

.conversation-list {
  border: none;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #fff;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
  text-align: center;
}

.empty-state h3 {
  margin: 20px 0 10px;
  color: #606266;
}

.empty-state p {
  margin: 0 0 30px;
  font-size: 14px;
}

.example-questions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
}

.messages-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.message {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.message-user {
  flex-direction: row-reverse;
}

.message-user .message-content {
  background-color: #409eff;
  color: white;
  border-radius: 18px 18px 4px 18px;
}

.message-assistant .message-content {
  background-color: #f4f4f5;
  color: #303133;
  border-radius: 18px 18px 18px 4px;
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  word-wrap: break-word;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 12px;
  opacity: 0.8;
}

.message-text {
  line-height: 1.6;
}

.message-text :deep(pre) {
  background-color: #f6f8fa;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 8px 0;
}

.message-text :deep(code) {
  background-color: #f6f8fa;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.message-sources {
  margin-top: 12px;
  font-size: 12px;
}

.sources-header {
  margin-bottom: 8px;
  font-weight: 600;
  color: #606266;
}

.source-item {
  border: 1px solid #ebeef5;
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
  transition: all 0.2s ease;
}

.source-item:hover {
  border-color: #c0c4cc;
}

.source-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background-color: #fafafa;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.source-header:hover {
  background-color: #f0f0f0;
}

.source-title-section {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.expand-icon {
  transition: transform 0.2s ease;
  color: #909399;
}

.expand-icon.expanded {
  transform: rotate(90deg);
}

.source-title-text {
  color: #409eff;
  text-decoration: none;
}

.source-content {
  padding: 12px;
  background-color: #fff;
  border-top: 1px solid #ebeef5;
}

.source-content-text {
  line-height: 1.6;
  color: #606266;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #909399;
}

.chat-input {
  border-top: 1px solid #e4e7ed;
  padding: 20px;
  background-color: #fff;
}

.input-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.input-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    display: none;
  }
  
  .message-content {
    max-width: 85%;
  }
}
</style>
