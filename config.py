from pydantic import SecretStr
from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path
import os
BASE_PATH = Path(__file__).parent

class Settings(BaseSettings):
    try:
        openai_api_key: SecretStr
        openai_api_base: str
        agent_model: str
        agent_temperature: float
        agent_max_tokens: int
        model_config = SettingsConfigDict(
            env_file = os.path.join(BASE_PATH,'.env'),
            env_file_encoding = 'utf-8',
            extra = 'ignore'
        )
    except:
        print(f'请检查{os.path.join(BASE_PATH,".env")}文件中的参数是否齐全')
if __name__ == '__main__':
    settings = Settings()
