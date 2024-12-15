from dataclasses import dataclass
from enum import Enum, unique
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_aws import ChatBedrock
import boto3, logging

logger = logging.getLogger(__name__)

@dataclass
class BaseModel:
    name: str
    tokenLimit: int
    
    def instance(self) -> BaseChatModel:
        pass

@dataclass
class BedrockModel(BaseModel):
    region: str
    model_id: str
    model_kwargs: dict

    def instance(self):
        logger.info(f"Creating Instance of BedrockModel {self.name}")
        instance = ChatBedrock(
            client=boto3.client('bedrock-runtime', region=self.region),
            model_id=self.model_id,
            model_kwargs=self.model_kwargs)
        return instance
    
@dataclass
class AzureModel(BaseModel):
    azure_endpoint: str
    openai_api_key: str
    openai_api_version: str
    azure_deployment: str
    temperature: float
    
    def instance(self):
        logger.info(f"Creating Instance of AzureModel {self.name}")
        instance = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            openai_api_key=self.openai_api_key,
            openai_api_version=self.openai_api_version,
            azure_deployment=self.azure_deployment,
            temperature=self.temperature)
        return instance
    
class GPT35Turbo(AzureModel):
    def __init__(self):
        super().__init__(
            name="GPT-35-Turbo",
            tokenLimit=1000,
            azure_endpoint="https://openaixpanseaisandbox.openai.azure.com",
            openai_api_key="0172189207fb4f08bd7fee99906b7b51",
            openai_api_version="2023-05-15",
            azure_deployment="gpt-35-turbo",
            temperature=0
            )

class Claude35Sonnet(BedrockModel):
    def __init__(self):
        super().__init__(
            name="Claude-35-Sonnet",
            tokenLimit=1000,
            region="us-east-1",
            model_id="anthropic.claude-35-sonnet-20240229-v1:0",
            model_kwargs={'temperature':0.5}
        )

@unique
class LLM(Enum):
    GPT_35_TURBO = GPT35Turbo
    CLAUDE_35_SONNET = Claude35Sonnet