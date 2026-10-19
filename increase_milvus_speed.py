import sys, os, time, json
sys.path.append(os.path.join(os.path.dirname(__file__), 'rags', 'medical_qa'))
import table_rags
from pymilvus import MilvusClient

client = MilvusClient(uri='http://localhost:19530')

# 1. 查看当前索引类型
print('索引信息:', client.list_indexes('medical_qa'))
for idx in client.list_indexes('medical_qa'):
    print(client.describe_index('medical_qa', idx))

# 2. 分解计时：embedding vs 纯向量检索
queries = ['感冒发烧怎么办', '高血压饮食注意什么', '糖尿病能治愈吗',
           '失眠怎么调理', '胃炎吃什么药', '骨折后如何康复',
           '头痛的原因有哪些', '小孩咳嗽老不好', '皮肤过敏怎么办', '颈椎病怎么缓解']

embed_times, search_times = [], []
for q in queries:
    t0 = time.perf_counter()
    vec = table_rags.EMBEDDINGS.embed_query(q)
    embed_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    client.search('medical_qa', data=[vec], limit=3, output_fields=['text'])
    search_times.append(time.perf_counter() - t0)

print(f'embedding 平均: {sum(embed_times)/len(embed_times)*1000:.0f}ms')
print(f'向量检索 平均: {sum(search_times)/len(search_times)*1000:.0f}ms  '
      f'最大: {max(search_times)*1000:.0f}ms')