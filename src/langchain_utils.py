from AzureConnection import AzureConnection
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)

# Model configuration
MODEL_CONFIG = {
    "azure_deployment": "DevGPT4o",
    "openai_api_version": "2024-02-01",
    "temperature": 0.2,
    "top_p": 0.9,
    "model": "DevGPT4o"
}

# Global variables
llm = None
conv = None
memory = None

def get_llm():
    """Get the configured LLM instance"""
    global llm
    if llm is None:
        # Get credentials from AzureConnection
        azure_conn = AzureConnection()
        
        # Create config with credentials
        llm_config = MODEL_CONFIG.copy()
        llm_config.update({
            "azure_endpoint": azure_conn.azure_endpoint,
            "api_key": azure_conn.api_key
        })
        
        llm = AzureChatOpenAI(**llm_config)
    return llm

def get_memory():
    """Get the configured memory instance"""
    global memory
    if memory is None:
        memory = ConversationBufferMemory()
    return memory

def get_conversation_chain():
    """Get the configured conversation chain"""
    global conv
    if conv is None:
        conv = ConversationChain(
            llm=get_llm(),
            memory=get_memory(),
            verbose=False
        )
    return conv

def create_conversation_chain_with_custom_memory(memory_type="buffer", **memory_kwargs):
    """Create conversation chain with custom memory settings"""
    if memory_type == "buffer":
        memory = ConversationBufferMemory(**memory_kwargs)
    elif memory_type == "window":
        k = memory_kwargs.pop("k", 5)
        memory = ConversationBufferWindowMemory(k=k, **memory_kwargs)
    elif memory_type == "summary":
        memory = ConversationSummaryMemory(llm=get_llm(), **memory_kwargs)
    elif memory_type == "token_buffer":
        max_token_limit = memory_kwargs.pop("max_token_limit", 2000)
        
        # Create a custom token buffer memory that handles token counting errors
        class RobustTokenBufferMemory(ConversationTokenBufferMemory):
            def save_context(self, inputs, outputs):
                """Override to handle token calculation errors gracefully"""
                try:
                    super().save_context(inputs, outputs)
                except NotImplementedError:
                    # If token counting fails, just add to buffer without pruning
                    self.chat_memory.add_user_message(str(inputs))
                    self.chat_memory.add_ai_message(str(outputs))
                except Exception as e:
                    # For any other errors, also just add to buffer
                    self.chat_memory.add_user_message(str(inputs))
                    self.chat_memory.add_ai_message(str(outputs))
        
        memory = RobustTokenBufferMemory(llm=get_llm(), max_token_limit=max_token_limit, **memory_kwargs)
    elif memory_type == "summary_buffer":
        max_token_limit = memory_kwargs.pop("max_token_limit", 2000)
        
        # Create a custom summary buffer memory that handles token counting errors
        class RobustSummaryBufferMemory(ConversationSummaryBufferMemory):
            def save_context(self, inputs, outputs):
                """Override to handle token calculation errors gracefully"""
                try:
                    super().save_context(inputs, outputs)
                except NotImplementedError:
                    # If token counting fails, just add to buffer without pruning
                    self.chat_memory.add_user_message(str(inputs))
                    self.chat_memory.add_ai_message(str(outputs))
                except Exception as e:
                    # For any other errors, also just add to buffer
                    self.chat_memory.add_user_message(str(inputs))
                    self.chat_memory.add_ai_message(str(outputs))
        
        memory = RobustSummaryBufferMemory(llm=get_llm(), max_token_limit=max_token_limit, **memory_kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
    
    return ConversationChain(
        llm=get_llm(),
        memory=memory,
        verbose=False
    )
