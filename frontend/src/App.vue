<script setup>
import { ref, nextTick, onMounted, onUnmounted } from 'vue'

const messages = ref([])
const input = ref('')
const loading = ref(false)
const userName = ref('')
const sessionId = ref('')
const sessions = ref([])          // 会话列表 [{id, title, updatedAt}]，updatedAt 降序
const dotCount = ref(0)
let dotTimer = null

const GREETING = '请问有什么可以帮到您的吗？'

// ---------- localStorage 持久化（前端级会话恢复） ----------
// ma_sessions：会话列表；ma_msgs_{session_id}：每个会话的消息（不含 GREETING，渲染时拼回）
const MSGS_KEY = (sid) => `ma_msgs_${sid}`
const SESSIONS_KEY = 'ma_sessions'

const loadMsgs = (sid) => {
  try { return JSON.parse(localStorage.getItem(MSGS_KEY(sid))) || [] } catch { return [] }
}
const saveMsgs = (sid, msgs) => {
  // GREETING 是渲染层拼接的欢迎语，不落存储
  const stored = msgs.filter(m => !(m.role === 'ai' && m.content === GREETING))
  try { localStorage.setItem(MSGS_KEY(sid), JSON.stringify(stored)) } catch { /* 容量超限则放弃本次持久化 */ }
}
const loadSessions = () => {
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY)) || [] } catch { return [] }
}
const saveSessions = () => {
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.value)) } catch { /* 同上 */ }
}
// 发生真实问答后：更新会话时间戳；首条提问同时作为会话标题
const touchSession = (sid, firstQuestion) => {
  const s = sessions.value.find(x => x.id === sid)
  if (!s) return
  s.updatedAt = Date.now()
  if (s.title === '新对话' && firstQuestion) s.title = firstQuestion.slice(0, 12)
  sessions.value.sort((a, b) => b.updatedAt - a.updatedAt)
  saveSessions()
}
// 一轮问答落本地：user + assistant 两条一起存
const persistTurn = (question, answer) => {
  const stored = loadMsgs(sessionId.value)
  stored.push({ role: 'user', content: question }, { role: 'ai', content: answer })
  saveMsgs(sessionId.value, stored)
  touchSession(sessionId.value, question)
}

// ---------- 会话操作 ----------
// 开启新会话：换新 session_id + 清空对话。
// session_id 变化后，后端 SessionManager 会为新 key 创建全新 Agent，历史上下文随之重置。
const newSession = () => {
  if (loading.value) return          // 流式请求进行中不允许切，避免孤儿消息
  sessionId.value = crypto.randomUUID()
  localStorage.setItem('session_id', sessionId.value)
  sessions.value.unshift({ id: sessionId.value, title: '新对话', updatedAt: Date.now() })
  saveSessions()
  messages.value = [{ role: 'ai', content: GREETING }]
}

// 切换到已有会话：从 localStorage 回显历史消息
const switchSession = async (sid) => {
  if (loading.value || sid === sessionId.value) return
  sessionId.value = sid
  localStorage.setItem('session_id', sid)
  messages.value = [{ role: 'ai', content: GREETING }, ...loadMsgs(sid)]
  await scrollToBottom()
}

// 删除会话：移出列表并清掉本地消息；删的是当前会话则切到最近一个
const removeSession = (sid) => {
  if (loading.value) return
  sessions.value = sessions.value.filter(s => s.id !== sid)
  localStorage.removeItem(MSGS_KEY(sid))
  saveSessions()
  if (sid === sessionId.value) {
    if (sessions.value.length) switchSession(sessions.value[0].id)
    else newSession()
  }
}

const startDots = () => {
  dotCount.value = 0
  dotTimer = setInterval(() => {
    dotCount.value = dotCount.value >= 3 ? 1 : dotCount.value + 1
  }, 400)
}
const stopDots = () => {
  if (dotTimer) {
    clearInterval(dotTimer)
    dotTimer = null
  }
}
onUnmounted(stopDots)

onMounted(async () => {
  userName.value = localStorage.getItem('user_name') || '用户' + Math.floor(Math.random() * 1000)
  sessionId.value = localStorage.getItem('session_id') || crypto.randomUUID()
  localStorage.setItem('user_name', userName.value)
  localStorage.setItem('session_id', sessionId.value)

  // 恢复会话列表；当前 sessionId 不在列表（旧数据/首次部署本功能）则补录
  sessions.value = loadSessions()
  if (!sessions.value.some(s => s.id === sessionId.value)) {
    sessions.value.unshift({ id: sessionId.value, title: '新对话', updatedAt: Date.now() })
  }
  sessions.value.sort((a, b) => b.updatedAt - a.updatedAt)
  saveSessions()

  // 恢复当前会话的历史消息
  messages.value = [{ role: 'ai', content: GREETING }, ...loadMsgs(sessionId.value)]
  await scrollToBottom()
})

const scrollToBottom = async () => {
  await nextTick()
  const el = document.querySelector('.messages')
  if (el) el.scrollTop = el.scrollHeight
}

const send = async () => {
  const question = input.value.trim()
  if (!question || loading.value) return

  messages.value.push({ role: 'user', content: question })
  input.value = ''

  const aiMsg = { role: 'ai', content: '' }
  messages.value.push(aiMsg)
  loading.value = true
  startDots()
  await scrollToBottom()

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_name: userName.value,
        session_id: sessionId.value,
        question,
      }),
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      aiMsg.content += decoder.decode(value, { stream: true })
      await scrollToBottom()
    }
  } catch (e) {
    aiMsg.content = `[请求失败] ${e.message}`
  } finally {
    stopDots()
    loading.value = false
    // 失败/空回复不持久化（从界面移除），避免污染恢复出来的历史
    if (!aiMsg.content || aiMsg.content.startsWith('[请求失败]')) {
      const idx = messages.value.indexOf(aiMsg)
      if (idx > -1) messages.value.splice(idx, 1)
    } else {
      persistTurn(question, aiMsg.content)
    }
    await scrollToBottom()
  }
}
</script>

<template>
  <div class="layout">
    <!-- 侧栏：会话列表（localStorage 持久化，刷新不丢） -->
    <aside class="sidebar">
      <button class="new-session" @click="newSession" :disabled="loading">+ 新会话</button>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          :class="['session-item', { active: s.id === sessionId }]"
          @click="switchSession(s.id)"
        >
          <span class="s-title">{{ s.title }}</span>
          <button class="s-del" title="删除会话" @click.stop="removeSession(s.id)" :disabled="loading">×</button>
        </div>
      </div>
    </aside>

    <div class="chat">
      <header class="chat-header">
        <span class="title">AI 医疗助手</span>
        <div class="header-right">
          <span class="session">会话 {{ sessionId.slice(0, 8) }}</span>
        </div>
      </header>

      <div class="messages">
        <div v-for="(msg, i) in messages" :key="i" :class="['msg', msg.role]">
          <div class="bubble">
            <template v-if="msg.content">{{ msg.content }}</template>
            <template v-else-if="msg.role === 'ai' && loading">
              思考中{{ '.'.repeat(dotCount) }}
            </template>
          </div>
        </div>
      </div>

      <div class="input-bar">
        <input
          v-model="input"
          placeholder="请输入你的问题…"
          @keyup.enter="send"
          :disabled="loading"
        />
        <button @click="send" :disabled="loading || !input.trim()">
          {{ loading ? '发送中' : '发送' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100vh;
}

/* ---------- 侧栏 ---------- */
.sidebar {
  width: 220px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #f0f2f5;
  border-right: 1px solid #e5e7eb;
}
.new-session {
  margin: 12px;
  padding: 9px 12px;
  background: #2c7be5;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.2s;
}
.new-session:hover:not(:disabled) { background: #1e5fc0; }
.new-session:disabled { opacity: 0.5; cursor: not-allowed; }

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 8px 12px;
}
.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  padding: 9px 10px;
  margin-bottom: 4px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  color: #333;
  transition: background 0.15s;
}
.session-item:hover { background: #e3e8ef; }
.session-item.active {
  background: #dce9fb;
  color: #1e5fc0;
  font-weight: 600;
}
.s-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.s-del {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  line-height: 18px;
  padding: 0;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: #9aa3af;
  font-size: 14px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s, background 0.15s;
}
.session-item:hover .s-del { opacity: 1; }
.s-del:hover { background: #d9534f; color: #fff; }

/* ---------- 主聊天区 ---------- */
.chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: #f5f6f8;
}

.chat-header {
  padding: 16px 20px;
  background: #2c7be5;
  color: #fff;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.chat-header .title { font-size: 18px; font-weight: 600; }
.chat-header .session { font-size: 12px; opacity: 0.8; }

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.msg { display: flex; }
.msg.user { justify-content: flex-end; }
.msg.ai { justify-content: flex-start; }

.bubble {
  max-width: 75%;
  padding: 10px 14px;
  border-radius: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 14px;
}
.msg.user .bubble {
  background: #2c7be5;
  color: #fff;
  border-bottom-right-radius: 4px;
}
.msg.ai .bubble {
  background: #fff;
  color: #333;
  border: 1px solid #e5e7eb;
  border-bottom-left-radius: 4px;
}

.input-bar {
  display: flex;
  gap: 8px;
  padding: 12px 20px;
  background: #fff;
  border-top: 1px solid #e5e7eb;
}
.input-bar input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
}
.input-bar input:focus { border-color: #2c7be5; }
.input-bar button {
  padding: 10px 20px;
  background: #2c7be5;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
}
.input-bar button:disabled {
  background: #9bb8e8;
  cursor: not-allowed;
}
</style>
