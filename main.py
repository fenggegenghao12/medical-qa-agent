from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
import sys
BASE_URL = Path(__file__).parent.resolve()
sys.path.append(os.path.join(BASE_URL,'rags/medical_qa'))
import langgraph
from typing import TypedDict
import tools
from config import Settings
from langchain_openai import ChatOpenAI
settings = Settings()
import tools
from langchain_core.messages import ToolMessage
import re
import json
from langchain.messages import HumanMessage,SystemMessage


def load_skill(skill_path=os.path.join(BASE_URL,'SKILL.md')):
    text = Path(skill_path).read_text(encoding='utf-8')
    if text.startswith('---'):
        text = text.split('---',2)[2].strip()
    return text

class Agent():
    def __init__(self,user_name,session_id):
        self.session_id = session_id
        self.user_name = user_name
        self.LLM = ChatOpenAI(
            base_url = settings.openai_api_base,
            api_key = settings.openai_api_key,
            model = settings.agent_model,
            temperature = settings.agent_temperature,
            max_tokens = settings.agent_max_tokens,
        )
        self.skill = load_skill()
        self.LLM = self.LLM.bind_tools([tools.query_answer,tools.get_query_time,tools.record_patient_information])
        self.messages = [{'role':'system','content':self.skill}]
        self.tool_map = {'query_answer':tools.query_answer,'get_query_time':tools.get_query_time,'record_patient_information':tools.record_patient_information}
    def run(self,question:str):
        role_messages = f'{self.user_name}的会话ID是{self.session_id},问题为{question}'
        self.messages.append({'role':'user','content':role_messages})
        for _ in range(5):
            response = self.LLM.invoke(self.messages)
            #print('<<<<response>>>>',response)
            if response.invalid_tool_calls:
                wrong_args = response.invalid_tool_calls[0]['args']
                try:
                    args = json.loads(wrong_args)
                except json.JSONDecodeError:
                    values = re.findall(r':(.*?)}',wrong_args)
                    values = [v.strip() for v in values]
                    keys = re.findall(r'"(.*?)"',wrong_args)
                    args = dict(zip(keys,values))
                    right_args = json.dumps(args,ensure_ascii=False)
                    # print('错误参数：',wrong_args)
                    # print('正确参数：',right_args)
                    response.tool_calls = [{'name':response.invalid_tool_calls[0]['name'],'args':right_args,'id':response.invalid_tool_calls[0]['id']}]
            if not response.tool_calls:
                return response.content
            self.messages.append(response)
            for tool_call in response.tool_calls:
                name = tool_call['name']
                args = tool_call['args']
                if isinstance(args,str):
                    args = json.loads(args)
                call_id = tool_call['id']
                result = self.tool_map[name].invoke(args)
                self.messages.append(ToolMessage(
                    content = str(result),
                    tool_call_id = call_id
                ))
        return '处理步数过多'

class SessionManager():
    def __init__(self):
        self._agents = {}
    def get_agent(self,user_name,session_id):
        if (user_name,session_id) not in self._agents:
            self._agents[(user_name,session_id)] = Agent(user_name,session_id)
        return self._agents[(user_name,session_id)]


if __name__ == '__main__':
    session_manager = SessionManager()
    agent = session_manager.get_agent(user_name='TOM',session_id='123456')
    print('------------------------如何检测孩子是否存在脑瘫？')
    print(agent.run('如何检测孩子是否存在脑瘫？'))
    print('------------------------原来是这样呀。')
    print(agent.run('原来是这样呀。'))
    print('------------------------那么我要预约，我叫张三，男性，18岁，电话是13880059121，症状是头痛。')
    print(agent.run('那么我要预约，我叫张三，男性，18岁，电话是13880059121，症状是头痛。'))
    print('------------------------好的，没有问题了。')
    print(agent.run('好的，没有问题了。'))
    agent = session_manager.get_agent(user_name='JACK',session_id='2222')
    print('------------------------请问可以给出建议吗？')
    print(agent.run('请问可以给出建议吗？'))