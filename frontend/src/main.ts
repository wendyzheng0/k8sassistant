import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import App from './App.vue'
import router from './router'

// 创建应用实例
const app = createApp(App)

// 使用插件 - 确保正确的初始化顺序
app.use(createPinia())
app.use(router)
app.use(ElementPlus)

// 注册 Element Plus 图标 - 移到插件注册之后
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 挂载应用
app.mount('#app')
