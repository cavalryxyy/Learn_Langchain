import os
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class AzureConnection:
    def __init__(self, api_key=None, azure_endpoint=None, api_version=None):
        # Priority: 1. Direct parameters, 2. Environment variables
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        self.client = None
        
        if not self.api_key or not self.azure_endpoint:
            raise ValueError("Azure OpenAI API key and endpoint must be provided or set as environment variables")

    def build_connection(self):
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        return self.client

# Create global embeddings instance for notebooks
try:
    azure_emb = AzureOpenAIEmbeddings(
        api_key=os.getenv('AZURE_EMBEDDING_API_KEY'),
        azure_endpoint=os.getenv('AZURE_EMBEDDING_ENDPOINT'),
        azure_deployment=os.getenv('AZURE_EMBEDDING_DEPLOYMENT', 'Embedding'),
        openai_api_version=os.getenv('AZURE_EMBEDDING_API_VERSION', '2023-05-15')
    )
    embeddings = azure_emb
    print('Embedding connection: connected')
except Exception as e:
    print(f'Embedding connection failed: {e}')
    embeddings = None
