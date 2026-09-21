import os
# 必须在 import langchain_ollama/httpx 之前执行：
# 1. 钉死 Ollama 地址，防止被环境变量 OLLAMA_HOST 劫持到随机端口
# 2. 清除代理，防止 trae-preview 终端的 HTTP_PROXY/ALL_PROXY 把本地请求转发走
os.environ['OLLAMA_HOST'] = 'http://127.0.0.1:11434'
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
for _k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    os.environ.pop(_k, None)
import shutil
import pandas as pd
from pathlib import Path
import pickle
import json
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import jieba
from langchain_classic.retrievers import EnsembleRetriever
from openai import OpenAI
from langchain_openai import ChatOpenAI
BASE_PATH = Path(__file__).parent.resolve()
import sys
sys.path.append(str(BASE_PATH.parent.parent))
from config import Settings
from cn_clip.clip import load_from_name
from cn_clip.clip import tokenize
from PIL import Image
import numpy as np
import torch
import faiss
import base64,io


model,preprocess = load_from_name('ViT-B-16',device='cpu',download_root=os.path.join(BASE_PATH,'models'))
pic_paths = []
all_vectors = []
all_paths = []

def _to_data_url(img):
    buf = io.BytesIO()
    img.save(buf,format='JPEG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return 'data:image/jpeg;base64,'+b64

def img2vectors(img_path):
    img = Image.open(img_path)
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        vector = model.encode_image(img)
        vector = vector/vector.norm(dim=1,keepdim=True)
    return vector

def all_image_paths(image_path=os.path.join(BASE_PATH,'images')):
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.endswith('.jpg'):
                all_paths.append(os.path.join(root,file))
    return all_paths

def generate_vectors():
    all_image_paths()
    for path in all_paths:
        vector = img2vectors(path)
        all_vectors.append(vector.cpu().numpy().flatten())
    vectors = np.array(all_vectors,dtype=np.float32)
    np.save(os.path.join(BASE_PATH,'vectors','image_vectors.npy'),vectors)
    return vectors

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index,os.path.join(BASE_PATH,'vectors','image_vectors_index.bin'))
    with open(os.path.join(BASE_PATH,'vectors','image_paths.json'),'w',encoding='utf-8') as f:
        json.dump(all_paths,f,ensure_ascii=False,indent=2)
    
def query_faiss_image(query,k=5):
    index = faiss.read_index(os.path.join(BASE_PATH,'vectors','image_vectors_index.bin'))
    image_path = os.path.join(BASE_PATH,'vectors','image_paths.json')
    with open(image_path,'r',encoding='utf-8') as f:
        image_paths = json.load(f)
    text_tokens = tokenize(query)
    with torch.no_grad():
        vector = model.encode_text(text_tokens)
        vector = vector/vector.norm(dim=1,keepdim=True)
    vector = vector.cpu().numpy().astype(np.float32)
    score,indices = index.search(vector,k)
    results_paths = []
    for i,j in enumerate(indices[0]):
        if 0 <= j < len(image_paths):
            results_paths.append(image_paths[j])
    return results_paths

EMBEDDINGS = OllamaEmbeddings(
    model='dengcao/bge-large-zh-v1.5:latest',
    base_url='http://127.0.0.1:11434',
    # trust_env=False：让底层 httpx 完全忽略环境里的代理/Trae端口转发，直连11434
    client_kwargs={'trust_env': False},
)
PERSIST_DIRECTORY = os.path.join(BASE_PATH,'chroma_db')
def json_rags(table='Chinese-medical-dialogue/data/query_answers.json'):
    documents = []
    count = 0
    with open(os.path.join(BASE_PATH,table),'r',encoding='utf-8') as f:
        data = json.load(f)
        for ele in data:
            count += 1
            documents.append(Document(page_content =ele.get('instruction','') +'：' + ele.get('input',''),metadata = {'output':ele.get('output','')}))
            if count == 500:
                print('-------document生成完毕--------')
                break
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        vectores = Chroma.from_documents(documents,EMBEDDINGS,persist_directory=PERSIST_DIRECTORY)
        print('-------vector生成完毕--------')
        f.close()
    return documents

def generate_retriever(vectores):
    bm25retriever = BM25Retriever.from_documents(vectores)
    bm25retriever.k = 5
    with open(os.path.join(BASE_PATH,'bm25/bm25.pkl'),'wb') as f:
        pickle.dump(vectores,f)
    print('-------bm25生成完毕--------')

def find_ensemble(query):
    with open(os.path.join(BASE_PATH,'bm25/bm25.pkl'),'rb') as f:
        vectores = pickle.load(f)
        f.close()
    bm25_retriever = BM25Retriever.from_documents(vectores,k=5,preprocess_func=jieba.lcut)
    vectores = Chroma(embedding_function=EMBEDDINGS,persist_directory=PERSIST_DIRECTORY)
    vector_retriever = vectores.as_retriever(search_kwargs={'k':5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,vector_retriever],weights=[0.2,0.8])
    results = ensemble_retriever.invoke(query)
    return results


def multi_answers_select(answers,k=5):
    answers_dict = {}
    for answer in answers:
        for rank,ele in enumerate(answer):
            if not ele in answers_dict.keys():
                answers_dict[ele] = []
            answers_dict[ele].append(rank+1)
    # print(answers_dict)
    scores_dict = {}
    func = lambda x: 1/(60 + x)
    for key in answers_dict.keys():
        scores_dict[key] = sum([func(rank) for rank in answers_dict[key]])
    sorted_scores_dict = dict(sorted(scores_dict.items(),key=lambda x:x[1],reverse=True))
    # print(sorted_scores_dict)
    return list(sorted_scores_dict.keys())[:k]
    
def query_answer(query):
    original_query = query
    settings = Settings()
    LLM = ChatOpenAI(
        api_key = settings.openai_api_key,
        base_url = settings.openai_api_base,
        model = settings.agent_model,
        temperature = settings.agent_temperature,
        max_tokens = settings.agent_max_tokens,
        model_kwargs={'response_format':{'type':'json_object'}}
    )
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
    subject_prompt = """
    你是一个专业的医疗助手，用户的问题是【{query}】需要你从中提取出问题的主题，例如：
    例1：
    问题：孩子太肥胖怎么办？
    主题：小儿肥胖
    返回结果格式：
    {{"subject":"小儿肥胖"}}
    例2：
    问题：如何检验孩子是否存在脑瘫？
    主题：小儿脑瘫
    返回结果格式：
    {{"subject":"小儿脑瘫"}}
    除此之外，请以JSON格式返回结果，不要包含markdown代码块或其他任何文字。
    """
    images_selection_prompt = """
    你是一个专业的从用户问题选择相匹配的图片助手，用户的主题是【{subject}】，你需要从【{images}】中选择出最符合用户问题的图片，如果用户提供的所有图片都不符合对应主题，则返回null，例如：
    例1：
    主题：小儿肥胖
    图片：
    1.图片1
    2.图片2
    3.图片3
    返回结果格式：
    {{"image":"1"}}
    例2：
    主题：小儿脑瘫
    图片：
    1.图片1
    2.图片2
    3.图片3
    4.图片4
    5.图片5
    返回结果格式：
    {{"image":"null"}}
    除此之外，请以JSON格式返回结果，不要包含markdown代码块或其他任何文字。
    """

    # images_selection_prompt = """
    # 你是一个专业的从用户问题选择相匹配的图片助手，用户的主题是【{subject}】，你需要从【{images}】中选择出最符合用户问题的图片，例如：
    # 主题：小儿肥胖
    # 图片：
    # 1.图片1
    # 2.图片2
    # 3.图片3
    # 返回结果格式：
    # {{"image":"1"}}
    # 除此之外，请以JSON格式返回结果，不要包含markdown代码块或其他任何文字。
    # """

    for i in range(5):
        try:
            sentences = LLM.invoke(sentences_prompt.format(query=query))
            json_content = json.loads(sentences.content)
            break
        except json.JSONDecodeError:
            print(f'用户问题解构中json解析错误第{i+1}次，重新尝试')
    querys = json_content.get('answers','')
    answers_list = []
    for query in querys:
        results = find_ensemble(query)
        ele_list = []
        for ele in results:
            ele_list.append(ele.metadata.get('output',''))
        answers_list.append(ele_list)
    res = multi_answers_select(answers_list)
    for i in range(5):
        try:
            selected_sentence = LLM.invoke(select_sentences_prompt.format(query=original_query,answers=res))
            answer = json.loads(selected_sentence.content).get('answer','')
            break
        except json.JSONDecodeError:
            print(f'选择结果中json解析错误第{i+1}次，重新尝试')
    for i in range(5):
        try:
            subject = LLM.invoke(subject_prompt.format(query=original_query))
            subject = json.loads(subject.content).get('subject','')
            break
        except json.JSONDecodeError:
            print(f'主题提取中json解析错误第{i+1}次，重新尝试')
    image_paths = query_faiss_image(subject)
    content = [{'type':'text','text':images_selection_prompt.format(subject=subject,images = '\n'.join(f'{i}.图片{i}' for i in range(1,len(image_paths)+1)))}]
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        content.append({'type':'image_url','image_url':{'url':_to_data_url(img)}})
    for i in range(5):
        try:
            selected_image = LLM.invoke([{'role':'user','content':content}])
            selected_image = json.loads(selected_image.content).get('image','')
            image = selected_image
            if selected_image != 'null':
                Image.open(image_paths[int(selected_image)-1]).show()
            else:
                print('没有符合主题的图片')
            break
        except json.JSONDecodeError:
            print(f'图片选择中json解析错误第{i+1}次，重新尝试')

    #return answer,image
    return answer


if __name__ == '__main__':
    # 建库只跑一次；完成后保持注释，日常只查询
    #vectores = json_rags()
    #generate_retriever(vectores)
    # answer,image = query_answer('孩子太胖了怎么办？')
    # print(answer)
    all_paths = all_image_paths()
    print(all_paths)
    vectors = generate_vectors()
    build_faiss_index(vectors)

    
    #print(results)