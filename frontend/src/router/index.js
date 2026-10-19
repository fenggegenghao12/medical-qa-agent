import { createRouter, createWebHistory } from 'vue-router'
import { h } from 'vue'

// 单页应用：真实界面由根组件 App.vue 直接渲染。
// 这里补一条根路径路由，避免 vue-router 报 "No match found for /" 警告。
// 用 render 函数而非 template 字符串：Vite 的 Vue 为 runtime-only 构建，不支持运行时模板编译。
const Blank = { render: () => h('div') }

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [{ path: '/', name: 'home', component: Blank }],
})

export default router
