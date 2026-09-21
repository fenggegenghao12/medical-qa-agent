import json, random
import os
from pathlib import Path
BASE_URL = os.path.join(Path(__file__).parent,'Chinese-medical-dialogue/data')
random.seed(42)
with open(os.path.join(BASE_URL,'query_answers.json'), encoding='utf-8') as f:
    data = json.load(f)

sample = random.sample(data, 100)
clean = []
for d in sample:
    if len(d['output']) < 10:          # 过滤过短回答
        continue
    clean.append({
        'instruction': '你是专业的医疗助手，请根据患者描述给出建议',
        'input': d['input'],
        'output': d['output'],
    })
with open(os.path.join(BASE_URL,'medical_100.json'), 'w', encoding='utf-8') as f:
    json.dump(clean, f, ensure_ascii=False, indent=2)