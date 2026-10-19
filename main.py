"""后端主入口：FastAPI 应用 + Agent 工具循环 + 空闲摘要 + 落库。

整体链路：
    前端 POST /chat
      → SessionManager 按 (user_name, session_id) 取/建 Agent
      → Agent.run() 手动工具循环（LLM 决策 → 调用 tools → ToolMessage 回填）
      → StreamingResponse 流式返回
      → finally 中写入 messages 表并重置 10 分钟空闲摘要定时器
"""
import os
import sys
import re
import json
import time
import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

BASE_URL = Path(__file__).parent.resolve()
sys.path.append(os.path.join(BASE_URL, 'rags/medical_qa'))

import tools
import storage
import table_rags
from config import Settings

settings = Settings()

app = FastAPI(title='AI医疗回复')


def init_logger():
    """初始化全局日志器：写入项目根目录的 log.txt，重复导入时避免重复挂 handler。"""
    logger = logging.getLogger("log")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler('log.txt', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


logger = init_logger()


@app.on_event("startup")
async def warmup():
    """服务启动预热：把首次请求才会触发的重活提前到启动时做，避免首条消息等待过久。

    依次预热：BM25 索引 → embedding 模型 → Milvus 连接 → SQLite 建表。
    """
    # 1. BM25 索引（bm25/bm25.pkl 存在则直接加载，否则现场构建）
    table_rags.get_bm25_retriever()
    print('[warmup] BM25 索引就绪')

    # 2. embedding 模型（发一次真实请求，把模型权重加载进内存）
    table_rags.EMBEDDINGS.embed_query('预热')
    print('[warmup] embedding 模型就绪')

    # 3. Milvus 连接（建一次连接，让底层连接池就绪）
    table_rags.get_vector_retriever()
    print('[warmup] Milvus 连接就绪')

    # 4. SQLite 建表（IF NOT EXISTS，幂等）
    storage.init_db()
    print('[warmup] 数据库就绪')


class ChatRequest(BaseModel):
    """POST /chat 的请求体结构。"""
    user_name: str
    session_id: str
    question: str


def load_skill(skill_path=os.path.join(BASE_URL, 'SKILL.md')):
    """读取 SKILL.md 作为 Agent 的 system prompt；若文件带 YAML front matter 则剥掉。"""
    text = Path(skill_path).read_text(encoding='utf-8')
    if text.startswith('---'):
        text = text.split('---', 2)[2].strip()
    return text


class Agent():
    """单个会话对应的 Agent：持有独立的对话历史、LLM（已绑定工具）与空闲摘要定时器。

    职责边界：
    - self.messages 是对话状态的唯一数据源（下标 0 是 system prompt，其余为 user/assistant/ToolMessage）
    - 记忆持久化只发生在两处：/chat 的 finally 写原文，_idle_summary 写摘要
    """

    def __init__(self, user_name, session_id):
        self.session_id = session_id
        self.user_name = user_name
        # 对话用 LLM：开启流式；随后 bind_tools 让模型知道有哪些工具可调
        self.LLM = ChatOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            model=settings.agent_model,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            streaming=True,
        )
        self._idle_timer = None
        self.skill = load_skill()
        self.LLM = self.LLM.bind_tools([tools.query_answer, tools.get_query_time, tools.record_patient_information])
        # 对话历史：下标 0 固定为 system prompt（人设 + 历史摘要），其余为真实对话
        self.messages = [{'role': 'system', 'content': self.skill}]
        # 工具名 → 工具函数的映射，run() 里按模型给出的 tool_call 名称分发
        self.tool_map = {'query_answer': tools.query_answer,
                         'get_query_time': tools.get_query_time,
                         'record_patient_information': tools.record_patient_information}
        self.summary = ''
        # 空闲摘要的提示词：要求把多轮对话与已有摘要合并成一句话，以 JSON 返回
        self.summary_prompt = """
        你是一个专业的医疗助手，善于将多轮对话总结成一句话，也善于把多轮对话与已有的历史摘要合并成一句话。
        要求：
        1. 必须保留对话中的关键信息，尤其是用户的个人信息（姓名、年龄、性别、症状、病史等），以及已经给出过的结论或建议。
        2. 只做压缩，不得添加对话中不存在的信息，也不能曲解原意。
        3. 若【历史摘要】为空，则只总结多轮对话；否则把两者合并，避免重复表述。
        4. 只输出一句话，语言简洁、专业。
        例如：
        多轮对话：
        用户：你好，我叫tom，今年20岁
        助手：你好tom，请问有什么可以帮到您？
        用户：我最近总是头晕，已经三天了
        历史摘要：（无）
        返回结果格式：
        {{"summary":"用户tom，20岁，自述头晕已持续三天。"}}
        又如：
        多轮对话：
        用户：对了，我还一直咳嗽
        历史摘要：用户tom，20岁，自述头晕已持续三天。
        返回结果格式：
        {{"summary":"用户tom，20岁，自述头晕持续三天，并伴有咳嗽。"}}
        除此之外，请以JSON格式返回结果，不要包含markdown代码块或其他任何文字。
        """
        self.max_turns = 10
        self._restore()

    def _restore(self):
        """从 instructions 表恢复历史摘要（若存在）。"""

        record = storage.get_instruction(self.user_name,self.session_id)
        if record:
            self.summary = record[3]
            #3为元组中instruction的下标
            self.messages[0]['content'] += f'\n历史摘要：{self.summary}'
        for m in storage.get_messages(self.user_name,self.session_id):
            self.messages.append({'role': m[3], 'content': m[4]})
            #3为元组中role的下标，4为元组中content的下标
    def reset_idle_timer(self, delay: int = 600):
        """重置空闲定时器：delay 秒（默认 600 = 10 分钟）无新消息则触发一次摘要。

        每次有新请求都会先 cancel 旧定时器再起新的，保证"最后一句话之后等满 10 分钟"。
        """
        if self._idle_timer:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(delay, self._idle_summary)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_summary(self):
        """空闲 10 分钟后的回调：把本轮对话与旧摘要合并，结果 upsert 进 instructions 表。

        注意：构造摘要输入时用 [system] + messages[1:] 拼新列表，
        绝不能直接把 messages[0] 覆盖成摘要提示词（会污染人设）。
        摘要成功后再把历史摘要追加回 system prompt，供后续对话引用。
        """
        try:
            if len(self.messages) > 1:
                summary = [{'role': 'system', 'content': f'{self.summary_prompt},摘要为{self.summary}'}] + self.messages[1:]
                response = table_rags.get_llm().invoke(summary)
                self.summary = json.loads(response.content)['summary']
                storage.save_instruction(self.user_name, self.session_id, self.summary)
                self.messages[0] = {'role': 'system', 'content': self.skill + f'\n历史摘要：{self.summary}'}
                logger.info(f'空闲总结完成 {self.user_name}的会话ID是{self.session_id},总结为：{self.summary}')
        except Exception as e:
            logger.error(f'空闲总结失败,错误信息：{e}')

    def run(self, question: str):
        """手动工具循环（生成器）：流式产出文本片段，模型给出工具调用则执行后继续循环。

        单次提问最多循环 5 轮，防止模型无限调用工具；模型不再给 tool_calls 时视为最终回答。
        """
        try:
            #role_messages = f'{self.user_name}的会话ID是{self.session_id},问题为{question}'
            self.messages.append({'role': 'user', 'content': question})
            for _ in range(5):
                final = None
                # 流式输出：chunk 既往前端 yield，也累加成完整的 final 消息
                for chunk in self.LLM.stream(self.messages):
                    if chunk.content:
                        yield chunk.content
                    final = chunk if final is None else final + chunk
                # 模型输出的工具参数 JSON 非法时，尝试用正则修复成合法 JSON 再执行
                if final.invalid_tool_calls:
                    wrong_args = final.invalid_tool_calls[0]['args']
                    try:
                        args = json.loads(wrong_args)
                    except json.JSONDecodeError:
                        values = re.findall(r':(.*?)}', wrong_args)
                        values = [v.strip() for v in values]
                        keys = re.findall(r'"(.*?)"', wrong_args)
                        args = dict(zip(keys, values))
                        right_args = json.dumps(args, ensure_ascii=False)
                        final.tool_calls = [{'name': final.invalid_tool_calls[0]['name'], 'args': right_args, 'id': final.invalid_tool_calls[0]['id']}]
                # 没有工具调用 → 这就是最终回答，落历史后结束
                if not final.tool_calls:
                    self.messages.append(final)
                    return
                self.messages.append(final)
                # 逐个执行工具调用，结果以 ToolMessage 回填历史，供下一轮 LLM 消化
                for tool_call in final.tool_calls:
                    name = tool_call['name']
                    args = tool_call['args']
                    if isinstance(args, str):
                        args = json.loads(args)
                    call_id = tool_call['id']
                    result = self.tool_map[name].invoke(args)
                    self.messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=call_id
                    ))
            yield '处理步数过多'
        except Exception as e:
            logger.error(f'处理问题时出错：{e}')
            yield '出现错误，暂时无法回答'


class SessionManager():
    """会话管理器：按 (user_name, session_id) 维护独立的 Agent 实例，实现会话隔离。"""

    def __init__(self):
        self._agents = {}

    def get_agent(self, user_name, session_id):
        """取指定用户的指定会话 Agent；不存在则新建。"""
        if (user_name, session_id) not in self._agents:
            self._agents[(user_name, session_id)] = Agent(user_name, session_id)
        return self._agents[(user_name, session_id)]


session_manager = SessionManager()


@app.post('/chat')
async def chat(request: ChatRequest):
    """聊天接口：流式返回回答；结束后把问答原文写入 messages 表并重置空闲定时器。"""
    print(f'[收到请求] user={request.user_name},session_id={request.session_id},question={request.question}')
    agent = session_manager.get_agent(request.user_name, request.session_id)

    def timed_run():
        """包装 Agent.run 的生成器：负责计时，并在 finally 中完成落库与定时器重置。"""
        answer = ''
        start = time.perf_counter()
        try:
            for chunk in agent.run(request.question):
                answer += chunk
                yield chunk
        finally:
            cost = time.perf_counter() - start
            logger.info(f'user={request.user_name} session={request.session_id} question={request.question} 耗时={cost:.1f}s')
            storage.save_message(request.user_name, request.session_id, 'user', request.question)
            storage.save_message(request.user_name, request.session_id, 'assistant', answer)
            agent.reset_idle_timer()
    return StreamingResponse(timed_run())
