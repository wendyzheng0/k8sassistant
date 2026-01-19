<template>
  <div class="admin-container">
    <!-- Header -->
    <header class="admin-header">
      <div class="header-left">
        <h1>
          <el-icon><Setting /></el-icon>
          向量数据库管理
        </h1>
        <el-tag type="info" size="large">
          共 {{ stats.total_chunks || 0 }} 个分块
        </el-tag>
      </div>
      <div class="header-right">
        <el-button 
          :icon="Refresh" 
          @click="refreshData"
          :loading="isLoading"
        >
          刷新
        </el-button>
        <router-link to="/">
          <el-button :icon="Back">返回聊天</el-button>
        </router-link>
      </div>
    </header>

    <!-- Main Content -->
    <div class="admin-content">
      <!-- Left Panel: Document Tree -->
      <aside class="tree-panel">
        <div class="panel-header">
          <h3>
            <el-icon><FolderOpened /></el-icon>
            文档结构
          </h3>
          <el-input
            v-model="searchKeyword"
            placeholder="搜索文档..."
            :prefix-icon="Search"
            clearable
            size="small"
            class="search-input"
          />
        </div>
        
        <div class="tree-container" v-loading="treeLoading">
          <el-tree
            ref="treeRef"
            :data="filteredTreeData"
            :props="treeProps"
            node-key="path"
            :expand-on-click-node="false"
            :default-expanded-keys="defaultExpandedKeys"
            highlight-current
            @node-click="handleNodeClick"
          >
            <template #default="{ node, data }">
              <span class="tree-node">
                <el-icon v-if="!data.is_file" class="folder-icon">
                  <Folder />
                </el-icon>
                <el-icon v-else class="file-icon">
                  <Document />
                </el-icon>
                <span class="node-label">{{ node.label }}</span>
              </span>
            </template>
          </el-tree>
          
          <el-empty 
            v-if="!treeLoading && filteredTreeData.length === 0" 
            description="暂无文档数据"
          />
        </div>
      </aside>

      <!-- Right Panel: Chunk Display -->
      <main class="chunks-panel">
        <div class="panel-header">
          <h3>
            <el-icon><Files /></el-icon>
            分块内容
          </h3>
          <div class="chunk-info" v-if="selectedPath">
            <el-tag type="success">
              {{ selectedPath }}
            </el-tag>
            <el-tag type="info">
              {{ chunks.length }} 个分块
            </el-tag>
          </div>
        </div>

        <div class="chunks-container" v-loading="chunksLoading">
          <div v-if="!selectedPath" class="empty-selection">
            <el-icon size="48"><DocumentCopy /></el-icon>
            <p>请从左侧选择一个文档查看其分块内容</p>
          </div>

          <div v-else-if="chunks.length === 0 && !chunksLoading" class="no-chunks">
            <el-icon size="48"><Warning /></el-icon>
            <p>该文档在向量数据库中没有分块数据</p>
            <p class="hint">可能该文档尚未被处理，或者路径不匹配</p>
          </div>

          <div v-else class="chunks-list">
            <div 
              v-for="(chunk, index) in chunks" 
              :key="chunk.id"
              class="chunk-card"
            >
              <div class="chunk-header">
                <div class="chunk-index">
                  <el-tag type="primary" effect="dark" size="small">
                    #{{ index + 1 }}
                  </el-tag>
                </div>
                <div class="chunk-meta">
                  <span class="chunk-id" :title="chunk.id">
                    ID: {{ truncateId(chunk.id) }}
                  </span>
                  <el-button
                    size="small"
                    :icon="CopyDocument"
                    @click="copyContent(chunk.content)"
                    circle
                  />
                </div>
              </div>
              <div class="chunk-content">
                <pre>{{ chunk.content }}</pre>
              </div>
              <div class="chunk-footer" v-if="chunk.metadata && Object.keys(chunk.metadata).length > 0">
                <el-collapse>
                  <el-collapse-item title="元数据" name="metadata">
                    <pre class="metadata-content">{{ formatMetadata(chunk.metadata) }}</pre>
                  </el-collapse-item>
                </el-collapse>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { 
  Setting, 
  Refresh, 
  Back, 
  FolderOpened, 
  Search, 
  Folder, 
  Document,
  Files,
  DocumentCopy,
  Warning,
  CopyDocument
} from '@element-plus/icons-vue'
import { ElMessage, ElTree } from 'element-plus'
import { adminAPI, type AdminTreeNode, type AdminChunk, type AdminStats } from '@/api/chat'

// Types (using imported types)
type TreeNode = AdminTreeNode
type ChunkInfo = AdminChunk
type Stats = AdminStats

// State
const treeRef = ref<InstanceType<typeof ElTree>>()
const treeData = ref<TreeNode[]>([])
const chunks = ref<ChunkInfo[]>([])
const stats = ref<Stats>({})
const selectedPath = ref<string>('')
const searchKeyword = ref<string>('')
const isLoading = ref(false)
const treeLoading = ref(false)
const chunksLoading = ref(false)

// Tree props
const treeProps = {
  children: 'children',
  label: 'label'
}

// Default expanded keys (first level)
const defaultExpandedKeys = computed(() => {
  return treeData.value.slice(0, 3).map(node => node.path)
})

// Filtered tree data based on search
const filteredTreeData = computed(() => {
  if (!searchKeyword.value) {
    return treeData.value
  }
  
  const keyword = searchKeyword.value.toLowerCase()
  
  function filterTree(nodes: TreeNode[]): TreeNode[] {
    return nodes.reduce((acc: TreeNode[], node) => {
      const matchesLabel = node.label.toLowerCase().includes(keyword)
      const matchesPath = node.path.toLowerCase().includes(keyword)
      
      if (node.children) {
        const filteredChildren = filterTree(node.children)
        if (filteredChildren.length > 0 || matchesLabel || matchesPath) {
          acc.push({
            ...node,
            children: filteredChildren.length > 0 ? filteredChildren : node.children
          })
        }
      } else if (matchesLabel || matchesPath) {
        acc.push(node)
      }
      
      return acc
    }, [])
  }
  
  return filterTree(treeData.value)
})

// Methods
async function loadDocumentTree() {
  treeLoading.value = true
  try {
    const response = await adminAPI.getDocumentTree()
    treeData.value = response.tree || []
  } catch (error: any) {
    console.error('Failed to load document tree:', error)
    ElMessage.error('加载文档树失败: ' + (error.message || '未知错误'))
  } finally {
    treeLoading.value = false
  }
}

async function loadStats() {
  try {
    const response = await adminAPI.getStats()
    stats.value = response
  } catch (error: any) {
    console.error('Failed to load stats:', error)
  }
}

async function loadChunks(filePath: string) {
  chunksLoading.value = true
  chunks.value = []
  
  try {
    const response = await adminAPI.getChunksByPath(filePath)
    chunks.value = response.chunks || []
  } catch (error: any) {
    console.error('Failed to load chunks:', error)
    ElMessage.error('加载分块失败: ' + (error.message || '未知错误'))
  } finally {
    chunksLoading.value = false
  }
}

function handleNodeClick(data: TreeNode) {
  if (data.is_file) {
    selectedPath.value = data.path
    loadChunks(data.path)
  }
}

async function refreshData() {
  isLoading.value = true
  try {
    await Promise.all([loadDocumentTree(), loadStats()])
    if (selectedPath.value) {
      await loadChunks(selectedPath.value)
    }
    ElMessage.success('刷新成功')
  } finally {
    isLoading.value = false
  }
}

function truncateId(id: string): string {
  if (id.length > 20) {
    return id.substring(0, 8) + '...' + id.substring(id.length - 8)
  }
  return id
}

function formatMetadata(metadata: Record<string, any>): string {
  return JSON.stringify(metadata, null, 2)
}

async function copyContent(content: string) {
  try {
    await navigator.clipboard.writeText(content)
    ElMessage.success('已复制到剪贴板')
  } catch (error) {
    ElMessage.error('复制失败')
  }
}

// Lifecycle
onMounted(() => {
  loadDocumentTree()
  loadStats()
})
</script>

<style scoped>
.admin-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(135deg, #1a1c2e 0%, #16213e 50%, #0f3460 100%);
  color: #e4e6eb;
  font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
}

.admin-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-left h1 {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #fff;
}

.header-right {
  display: flex;
  gap: 12px;
}

.admin-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Tree Panel */
.tree-panel {
  width: 360px;
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.03);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
}

.panel-header {
  padding: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.panel-header h3 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.search-input {
  width: 100%;
}

.search-input :deep(.el-input__wrapper) {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: none;
}

.search-input :deep(.el-input__inner) {
  color: #e4e6eb;
}

.search-input :deep(.el-input__inner::placeholder) {
  color: #64748b;
}

.tree-container {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.tree-container :deep(.el-tree) {
  background: transparent;
  color: #e4e6eb;
  --el-tree-node-hover-bg-color: rgba(59, 130, 246, 0.15);
  --el-tree-node-content-height: 36px;
}

.tree-container :deep(.el-tree-node__content:hover) {
  background: rgba(59, 130, 246, 0.15);
}

.tree-container :deep(.el-tree-node.is-current > .el-tree-node__content) {
  background: rgba(59, 130, 246, 0.25);
  border-left: 3px solid #3b82f6;
}

.tree-node {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}

.folder-icon {
  color: #fbbf24;
}

.file-icon {
  color: #60a5fa;
}

.node-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Chunks Panel */
.chunks-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chunks-panel .panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.chunk-info {
  display: flex;
  gap: 8px;
}

.chunks-container {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.empty-selection,
.no-chunks {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #64748b;
  text-align: center;
}

.empty-selection p,
.no-chunks p {
  margin: 12px 0 0;
  font-size: 14px;
}

.no-chunks .hint {
  font-size: 12px;
  opacity: 0.7;
}

.chunks-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chunk-card {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  overflow: hidden;
  transition: all 0.2s ease;
}

.chunk-card:hover {
  border-color: rgba(59, 130, 246, 0.4);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.chunk-index {
  display: flex;
  align-items: center;
  gap: 8px;
}

.chunk-meta {
  display: flex;
  align-items: center;
  gap: 12px;
}

.chunk-id {
  font-size: 11px;
  color: #64748b;
  font-family: 'JetBrains Mono', monospace;
}

.chunk-content {
  padding: 16px;
  max-height: 400px;
  overflow-y: auto;
}

.chunk-content pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 13px;
  line-height: 1.6;
  color: #cbd5e1;
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
}

.chunk-footer {
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.chunk-footer :deep(.el-collapse) {
  border: none;
}

.chunk-footer :deep(.el-collapse-item__header) {
  background: rgba(255, 255, 255, 0.03);
  color: #94a3b8;
  font-size: 12px;
  padding: 0 16px;
  border: none;
}

.chunk-footer :deep(.el-collapse-item__wrap) {
  background: transparent;
  border: none;
}

.chunk-footer :deep(.el-collapse-item__content) {
  padding: 12px 16px;
  color: #94a3b8;
}

.metadata-content {
  margin: 0;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.25);
}

/* Element Plus overrides for dark theme */
:deep(.el-button) {
  --el-button-bg-color: rgba(255, 255, 255, 0.1);
  --el-button-border-color: rgba(255, 255, 255, 0.2);
  --el-button-text-color: #e4e6eb;
  --el-button-hover-bg-color: rgba(255, 255, 255, 0.15);
  --el-button-hover-border-color: rgba(255, 255, 255, 0.3);
  --el-button-hover-text-color: #fff;
}

:deep(.el-button--primary) {
  --el-button-bg-color: #3b82f6;
  --el-button-border-color: #3b82f6;
  --el-button-hover-bg-color: #2563eb;
  --el-button-hover-border-color: #2563eb;
}

:deep(.el-tag) {
  border: none;
}

:deep(.el-loading-mask) {
  background-color: rgba(26, 28, 46, 0.8);
}

:deep(.el-loading-spinner .circular) {
  stroke: #3b82f6;
}

:deep(.el-empty__description p) {
  color: #64748b;
}

/* Responsive */
@media (max-width: 1024px) {
  .tree-panel {
    width: 280px;
  }
}

@media (max-width: 768px) {
  .admin-content {
    flex-direction: column;
  }
  
  .tree-panel {
    width: 100%;
    height: 40%;
    border-right: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .chunks-panel {
    height: 60%;
  }
}
</style>

