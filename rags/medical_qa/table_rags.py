"""RAG 核心管线：混合检索（BM25 + Milvus 向量）→ RRF 融合 → LLM 选句重写。

运行时只保留查询链路；建库脚本（json_rags / generate_retriever）与
图像检索（CN-CLIP + FAISS）、数据预处理（get_theme / theme_chunk）均未随项目迁移。
"""
import os
import json
import pickle
from pathlib import Path
import sys

# 必须在 import langchain_huggingface 之前设置：
# 模型已完整缓存在本地，禁止 huggingface_hub 发任何网络请求（否则国内直连超时反复重试）
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import jieba
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
from langchain_core.documents import Document

BASE_PATH = Path(__file__).parent.resolve()
sys.path.append(str(BASE_PATH.parent.parent))
from config import Settings

MILVUS_HOST = os.getenv('MILVUS_HOST', '127.0.0.1')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
COLLECTION_NAME = 'medical_qa'

# 模块级单例缓存：BM25 检索器 / Milvus 向量检索器 / 摘要用 LLM，全进程只建一次
_bm25_cache = None
_vector_retriever = None
_LLM_CACHE = None

# embedding 模型：BGE-small-zh（512 维，COSINE），本地 CPU 推理
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-zh-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def get_llm():
    """返回摘要/选句共用的 LLM 单例（带 JSON 输出模式，便于 json.loads 解析）。"""
    global _LLM_CACHE
    if _LLM_CACHE is None:
        settings = Settings()
        _LLM_CACHE = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.agent_model,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            model_kwargs={'response_format': {'type': 'json_object'}},
        )
    return _LLM_CACHE

def json_rags(table='Chinese-medical-dialogue/data/query_answers.json'):
    documents = []
    ids = []
    count = 0
    batch_size = 512
    with open(os.path.join(BASE_PATH,table),'r',encoding='utf-8') as f:
        data = json.load(f)
        for ele in data:
            count += 1
            page_content = ele.get('instruction','') +'：' + ele.get('input','')
            documents.append(Document(page_content=page_content,metadata = {'output':ele.get('output','')}))
            ids.append(f"{count-1}_{hashlib.md5(page_content.encode('utf-8')).hexdigest()}")
            if count % 10000 == 0:
                print(f'-------{count}个document生成完毕--------')
        # 不删库：目录存在则打开旧库追加，不存在则自动创建
        vectore_store = Milvus(embedding_function=EMBEDDINGS,connection_args={'host':MILVUS_HOST,'port':MILVUS_PORT},collection_name=COLLECTION_NAME,auto_id=False) 
        for i in range(0,len(documents),batch_size):
            tiny_batch = batch_size
            start = i
            for retry in range(3):
                try:
                    end = i + batch_size
                    for j in range(start,end,tiny_batch):
                        start = j
                        vectore_store.add_documents(documents[j:j+tiny_batch],ids=ids[j:j+tiny_batch])
                        print(f'已经入库{j+tiny_batch}个document')
                    break
                except Exception as e:
                    print(f'[异常] {e}')
                    print(f'第{retry+1}次重新入库{i+batch_size}个document')
                    tiny_batch = tiny_batch // 2
                    time.sleep(1)
        print('-------vector生成完毕--------')
    return documents

def get_bm25_retriever():
    """返回 BM25 检索器单例：优先从 bm25/bm25.pkl 加载文档集并现场构建索引。

    注意：bm25.pkl 随项目一起迁移，必须存在；若缺失会直接 FileNotFoundError，
    重建需用建库脚本（未随项目迁移）。
    """
    global _bm25_cache
    if _bm25_cache is None:
        pkl = os.path.join(BASE_PATH, 'bm25/bm25.pkl')
        if os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                docs = pickle.load(f)
            _bm25_cache = BM25Retriever.from_documents(docs, k=3, preprocess_func=jieba.lcut)
        else:
            with open(pkl, 'rb') as f:
                docs = pickle.load(f)
            _bm25_cache = BM25Retriever.from_documents(docs, k=3, preprocess_func=jieba.lcut)
            with open(pkl, 'wb') as f:
                pickle.dump(_bm25_cache, f)
    return _bm25_cache


def get_vector_retriever():
    """返回 Milvus 向量库连接单例（medical_qa collection）。"""
    global _vector_retriever
    if _vector_retriever:
        return _vector_retriever
    else:
        _vector_retriever = Milvus(embedding_function=EMBEDDINGS,
                                   connection_args={'host': MILVUS_HOST, 'port': MILVUS_PORT},
                                   collection_name=COLLECTION_NAME)
        return _vector_retriever


def find_ensemble(query):
    """对单个查询做混合检索：BM25(权重0.2) + 向量(权重0.8)，EnsembleRetriever 内部用 RRF 融合。"""
    bm25_retriever = get_bm25_retriever()
    vectores = get_vector_retriever()
    vector_retriever = vectores.as_retriever(search_kwargs={'k': 3})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.2, 0.8])
    results = ensemble_retriever.invoke(query)
    return results


def multi_answers_select(answers, k=3):
    """多路检索结果做 RRF 融合排序，返回得分最高的前 k 条答案。

    :param answers: 每路检索的答案列表（列表的列表），答案按各路内部排名计分
    :param k: 返回条数
    """
    answers_dict = {}
    for answer in answers:
        for rank, ele in enumerate(answer):
            if not ele in answers_dict.keys():
                answers_dict[ele] = []
            answers_dict[ele].append(rank + 1)
    # RRF 公式：score = Σ 1/(60+rank)，排名越靠前、被越多路命中得分越高
    scores_dict = {}
    func = lambda x: 1 / (60 + x)
    for key in answers_dict.keys():
        scores_dict[key] = sum([func(rank) for rank in answers_dict[key]])
    sorted_scores_dict = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))
    return list(sorted_scores_dict.keys())[:k]


def query_answer(query):
    """完整 RAG 问答：查询改写×3 → 并行混合检索 → RRF 融合 → LLM 选句重写。

    :param query: 用户的医疗问题
    :return: 重写后的最终答案字符串
    """
    original_query = query
    LLM = get_llm()
    max_turns = 10
    # 提示词1：把用户问题拆成 3 句同义问，提升多路召回
    sentences_prompt = """
    你是一个专业的医疗助手，你可以将用户询问的问题【{query}】分解为3句意思完全相同的句子。
    例如：
    问题：孩子太肥胖怎么办？
    解析：
    1.孩子太肥胖了，怎么办？
    2.应该如何让孩子减肥？
    3.孩子太胖，如何应对？
    返回结果格式：
    {{"answers":["1.孩子太肥胖了，怎么办？","2.应该如何让孩子减肥？","3.孩子太胖，如何应对？"]}}
    除此之外，请以JSON格式返回结果，不要包含markdown代码块或其他任何文字
    """
    # 提示词2：从融合检索结果中选出最贴合原问题的一句，并礼貌地重写
    select_sentences_prompt = """
    你是一个专业的医疗助手，用户询问的问题是【{query}】需要你从【{answers}】中选择一句最符合用户问题的结果，并对其进行适当的修改，但是一定不能够曲解所选择的句子的意思，以符合当前的语境，要保持礼貌和专业。
    例如：
    问题：孩子太胖怎么办？
    选择：
    1.多运动，少吃饭
    2.控制饮食，避免过量
    3.保持良好的睡眠习惯
    返回结果格式：
    {{"answer":"您好，孩子需要多运动，少吃饭哦"}}
    除此之外，请以JSON格式返回结果，不要包含markdown代码块或其他任何文字。
    """
    # 1. 查询改写：JSON 解析失败最多重试 5 次
    for i in range(5):
        try:
            sentences = LLM.invoke(sentences_prompt.format(query=query))
            json_content = json.loads(sentences.content)
            break
        except json.JSONDecodeError:
            print(f'用户问题解构中json解析错误第{i+1}次，重新尝试')
    querys = json_content.get('answers', '')

    # 2. 三路同义问并行检索，每路提取答案文本列表
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(find_ensemble, query) for query in querys]
        answers_list = []
        for future in futures:
            results = future.result()
            ele_list = [ele.metadata.get('output', '') for ele in results]
            answers_list.append(ele_list)

    # 3. RRF 融合取 top3，再让 LLM 选一句重写成最终答案
    res = multi_answers_select(answers_list)
    for i in range(5):
        try:
            selected_sentence = LLM.invoke(select_sentences_prompt.format(query=original_query, answers=res))
            answer = json.loads(selected_sentence.content).get('answer', '')
            break
        except json.JSONDecodeError:
            print(f'选择结果中json解析错误第{i+1}次，重新尝试')
    return answer
