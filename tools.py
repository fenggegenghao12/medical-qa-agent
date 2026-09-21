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
    name: str
    age: int
    gender: str
    phone: str
    symptom: str = ''
    medical_interview: str = '否'
    query_time: str = ''

@tool
def query_answer(question:str) -> str:
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
def record_patient_information(patient_information:PatientInformation) -> bool:
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
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(patient_information.model_dump())
        return True
    except Exception:
        return False
