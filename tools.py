"""LLM 工具定义：每个 @tool 函数的签名和 docstring 就是给模型看的调用说明。

原则：工具必须是纯函数，签名里只放模型能填的参数；
不把 history/summary 之类的运行时状态塞进签名（模型只会瞎编）。
"""
import os
import csv
import sys
import datetime
from pathlib import Path
from pydantic import BaseModel

BASE_URL = Path(__file__).resolve().parent
sys.path.append(os.path.join(BASE_URL, 'rags', 'medical_qa'))

# 别名导入：避免和下面的工具函数同名遮蔽（否则工具内部会递归调用自己）
from table_rags import query_answer as rag_query_answer
from langchain_core.tools import tool


class PatientInformation(BaseModel):
    """record_patient_information 的参数 Schema：显式声明字段，让模型知道要填什么。"""
    name: str
    age: int
    gender: str
    phone: str
    symptom: str = ''
    medical_interview: str = '否'
    query_time: str = ''


@tool
def query_answer(question: str) -> str:
    """基于本地医疗RAG知识库回答用户的医疗问题。
    当用户咨询疾病、症状、用药、护理等医疗健康问题时调用。

    :param question: 患者询问的医疗问题
    :return: 医疗问题的答案
    """
    return rag_query_answer(question)


@tool
def get_query_time() -> str:
    """获取用户咨询时间。在记录患者信息前调用。
    :return: 咨询时间字符串，格式 YYYY-MM-DD HH:MM:SS
    """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@tool
def record_patient_information(patient_information: PatientInformation) -> bool:
    """记录患者医疗相关信息到本地CSV档案。
    在获取咨询时间、且患者信息齐全后调用。
    :param patient_information: 患者信息字典，字段：
        name(姓名)、age(年龄)、gender(性别)、phone(手机号)、
        symptom(症状)、medical_interview(是否预约)、query_time(咨询时间)
    :return: True 记录成功 / False 记录失败
    """
    try:
        csv_path = os.path.join(BASE_URL, 'patients_information', 'informations.csv')
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['name', 'age', 'gender', 'phone',
                      'symptom', 'medical_interview', 'query_time']
        # 空文件/新文件需要先写表头；utf-8-sig 让 Excel 打开不乱码
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(patient_information.model_dump())
        return True
    except Exception:
        return False
