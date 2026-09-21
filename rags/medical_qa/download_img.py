from bing_image_downloader import downloader
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent

queries = [
    '扁桃体炎', '中耳炎', '婴幼儿腹泻',
    '小儿肥胖', '小儿脑瘫', '新生儿黄疸','新生儿吸入性肺炎','婴幼儿便秘','婴幼儿缺钙'
]

for q in queries:
    #os.makedirs(os.path.join(BASE_DIR,'images',q), exist_ok=True)
    downloader.download(q, limit=20, output_dir=os.path.join(BASE_DIR,'images'), 
                       adult_filter_off=True, force_replace=False)