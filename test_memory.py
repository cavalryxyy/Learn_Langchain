import os
from AzureConnection import AzureConnection
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationChain, SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)

# Centralized model configuration
MODEL_CONFIG = {
    "azure_deployment": "DevGPT4o",
    "openai_api_version": "2024-02-01",
    "azure_endpoint": "https://azswcdevbktchatgpt.openai.azure.com/",
    "api_key": "3c33f09702264798ba722c2bc0f78571",  # Changed from azure_api_key to api_key
    "temperature": 0.2,
    "top_p": 0.9,
    "model": "DevGPT4o"  # Keep your original model name
}

# Global variables for reuse
llm = None
conv = None
memory = None

def setup_azure_connection():
    """Set up Azure OpenAI connection"""
    # Initialize the Azure connection
    azure_conn = AzureConnection(
        api_key="3c33f09702264798ba722c2bc0f78571",
        azure_endpoint="https://azswcdevbktchatgpt.openai.azure.com/",
        api_version="2024-02-01"
    )
    
    # Build the client
    client = azure_conn.build_connection()
    print('Azure connection: connected')
    return client

def setup_llm():
    """Set up Azure OpenAI LLM with centralized configuration"""
    global llm
    
    # Create a custom LLM class that handles token counting for custom model names
    class CustomAzureChatOpenAI(AzureChatOpenAI):
        def get_num_tokens_from_messages(self, messages):
            """Override to handle custom model names like DevGPT4o"""
            try:
                return super().get_num_tokens_from_messages(messages)
            except (KeyError, Exception):
                # Fallback to approximate token counting for custom models
                total_tokens = 0
                for message in messages:
                    try:
                        if hasattr(message, 'content'):
                            content = message.content
                        elif hasattr(message, 'text'):
                            content = message.text
                        else:
                            content = str(message)
                        
                        # Rough approximation: 1 token ¡Ö 4 characters
                        if content:
                            total_tokens += len(content) // 4
                    except Exception:
                        # If we can't process a message, skip it
                        continue
                
                return max(total_tokens, 1)  # Ensure we return at least 1 token
    
    llm = CustomAzureChatOpenAI(**MODEL_CONFIG)
    return llm

def setup_memory():
    """Set up conversation memory"""
    global memory
    memory = ConversationBufferMemory()
    return memory

def setup_conversation_chain():
    """Set up conversation chain with Azure OpenAI and memory"""
    global conv, llm, memory
    
    # Set up LLM if not already done
    if llm is None:
        llm = setup_llm()
    
    # Set up memory if not already done
    if memory is None:
        memory = setup_memory()
    
    # Conversation chain combines LLM and memory
    conv = ConversationChain(
        llm=llm,  # The Azure OpenAI model
        memory=memory,  # Conversation history storage
        verbose=False
    )
    return conv

def get_llm():
    """Get the configured LLM instance"""
    global llm
    if llm is None:
        llm = setup_llm()
    return llm

def get_conversation_chain():
    """Get the configured conversation chain"""
    global conv
    if conv is None:
        conv = setup_conversation_chain()
    return conv

def get_memory():
    """Get the configured memory instance"""
    global memory
    if memory is None:
        memory = setup_memory()
    return memory

def create_memory(memory_type="buffer", **kwargs):
    """
    Create memory with custom parameters
    
    Args:
        memory_type (str): Type of memory to create
            - "buffer": ConversationBufferMemory (default)
            - "window": ConversationBufferWindowMemory
            - "summary": ConversationSummaryMemory
            - "token_buffer": ConversationTokenBufferMemory
            - "summary_buffer": ConversationSummaryBufferMemory
        **kwargs: Additional parameters for the memory type
    
    Returns:
        Memory instance
    """
    if memory_type == "buffer":
        return ConversationBufferMemory(**kwargs)
    
    elif memory_type == "window":
        # Default window size is 5, but you can override with k parameter
        k = kwargs.pop("k", 5)
        return ConversationBufferWindowMemory(k=k, **kwargs)
    
    elif memory_type == "summary":
        # Requires llm for summarization
        llm = get_llm()
        return ConversationSummaryMemory(llm=llm, **kwargs)
    
    elif memory_type == "token_buffer":
        # Default max token limit is 2000
        max_token_limit = kwargs.pop("max_token_limit", 2000)
        llm = get_llm()
        
        # Create a more robust token buffer memory
        class RobustTokenBufferMemory(ConversationTokenBufferMemory):
            def save_context(self, inputs, outputs):
                """Override to handle token calculation errors gracefully"""
                try:
                    super().save_context(inputs, outputs)
                except Exception as e:
                    # If token calculation fails, just add to buffer without pruning
                    self.chat_memory.add_user_message(str(inputs))
                    self.chat_memory.add_ai_message(str(outputs))
        
        return RobustTokenBufferMemory(llm=llm, max_token_limit=max_token_limit, **kwargs)
    
    elif memory_type == "summary_buffer":
        # Combines summary and buffer
        llm = get_llm()
        max_token_limit = kwargs.pop("max_token_limit", 2000)
        return ConversationSummaryBufferMemory(
            llm=llm, 
            max_token_limit=max_token_limit, 
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

def create_conversation_chain_with_custom_memory(memory_type="buffer", **memory_kwargs):
    """
    Create conversation chain with custom memory settings
    
    Args:
        memory_type (str): Type of memory to use
        **memory_kwargs: Parameters for the memory
    
    Returns:
        ConversationChain instance
    """
    llm = get_llm()
    memory = create_memory(memory_type, **memory_kwargs)
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

def test_conversation():
    """Test the conversation chain with memory"""
    print("Setting up Azure connection...")
    client = setup_azure_connection()
    
    print("\nSetting up conversation chain...")
    conv = get_conversation_chain()
    
    print("\n=== Testing Conversation Chain with Your Model Config ===")
    print(f"Model: {MODEL_CONFIG['model']}")
    print(f"Temperature: {MODEL_CONFIG['temperature']}")
    print(f"Top_p: {MODEL_CONFIG['top_p']}")
    
    # Test 1: Initial greeting
    print("\nTest 1: Initial greeting")
    response1 = conv.predict(input="Hello! How are you?")
    print(f"AI Response: {response1}")
    
    # Test 2: Memory test - asking about previous conversation
    print("\nTest 2: Memory test")
    response2 = conv.predict(input="What did I just ask you?")
    print(f"AI Response: {response2}")
    
    # Test 3: Continue conversation
    print("\nTest 3: Continue conversation")
    response3 = conv.predict(input="Can you tell me about Python programming?")
    print(f"AI Response: {response3}")
    
    # Test 4: Memory test again
    print("\nTest 4: Memory test - asking about Python")
    response4 = conv.predict(input="What did I ask you about?")
    print(f"AI Response: {response4}")
    
    # Test 5: Check memory contents
    print("\nTest 5: Memory contents")
    print("Memory buffer:")
    print(conv.memory.buffer)

def test_with_system_instruction():
    """Test with system instruction like your Azure client"""
    print("\n=== Testing with System Instruction ===")
    
    # Your system instruction
    system_instruction = "You are a helpful AI assistant. Be concise and accurate in your responses."
    
    # Use the centralized LLM
    llm = get_llm()
    
    # Test direct LLM call with system message
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = llm.invoke(messages)
    print(f"Direct LLM Response: {response.content}")

def test_different_memory_types():
    """Test different types of memory with examples"""
    print("\n=== Testing Different Memory Types ===")
    
    # Test 1: ConversationBufferMemory (default)
    print("\n1. ConversationBufferMemory (stores all messages)")
    conv_buffer = create_conversation_chain_with_custom_memory("buffer")
    conv_buffer.predict(input="Hello! My name is Alice.")
    conv_buffer.predict(input="What's my name?")
    print(f"Buffer Memory Messages: {len(conv_buffer.memory.chat_memory.messages)}")
    
    # Test 2: ConversationBufferWindowMemory (stores last k messages)
    print("\n2. ConversationBufferWindowMemory (stores last 3 messages)")
    conv_window = create_conversation_chain_with_custom_memory("window", k=3)
    conv_window.predict(input="Message 1")
    conv_window.predict(input="Message 2")
    conv_window.predict(input="Message 3")
    conv_window.predict(input="Message 4")
    conv_window.predict(input="How many messages do you remember?")
    print(f"Window Memory Messages: {len(conv_window.memory.chat_memory.messages)}")
    
    # Test 3: ConversationTokenBufferMemory (token-based limit)
    print("\n3. ConversationTokenBufferMemory (token limit: 100)")
    conv_token = create_conversation_chain_with_custom_memory("token_buffer", max_token_limit=100)
    conv_token.predict(input="This is a short message.")
    conv_token.predict(input="This is another short message.")
    conv_token.predict(input="What do you remember?")
    print(f"Token Buffer Memory Messages: {len(conv_token.memory.chat_memory.messages)}")
    
    # Test 4: ConversationSummaryMemory (summarizes old messages)
    print("\n4. ConversationSummaryMemory (summarizes old messages)")
    conv_summary = create_conversation_chain_with_custom_memory("summary")
    conv_summary.predict(input="I love programming in Python.")
    conv_summary.predict(input="Python is great for data science.")
    conv_summary.predict(input="What did we talk about?")
    print(f"Summary Memory: {conv_summary.memory.moving_summary_buffer}")
    
    # Test 5: ConversationSummaryBufferMemory (combines summary + buffer)
    print("\n5. ConversationSummaryBufferMemory (summary + recent buffer)")
    conv_summary_buffer = create_conversation_chain_with_custom_memory(
        "summary_buffer", 
        max_token_limit=200
    )
    conv_summary_buffer.predict(input="I work as a software engineer.")
    conv_summary_buffer.predict(input="I use Python and JavaScript.")
    conv_summary_buffer.predict(input="What's my job?")
    print(f"Summary Buffer Memory: {conv_summary_buffer.memory.moving_summary_buffer}")

def demonstrate_memory_parameters():
    """Demonstrate different memory parameter configurations"""
    print("\n=== Memory Parameter Examples ===")
    
    # Example 1: Window memory with custom window size
    print("\nExample 1: Window memory with k=2")
    conv_window_2 = create_conversation_chain_with_custom_memory("window", k=2)
    conv_window_2.predict(input="First message")
    conv_window_2.predict(input="Second message")
    conv_window_2.predict(input="Third message")
    conv_window_2.predict(input="How many messages do you remember?")
    
    # Example 2: Token buffer with custom token limit
    print("\nExample 2: Token buffer with 50 token limit")
    conv_token_50 = create_conversation_chain_with_custom_memory("token_buffer", max_token_limit=50)
    conv_token_50.predict(input="Short message.")
    conv_token_50.predict(input="Another short message.")
    conv_token_50.predict(input="What do you remember?")
    
    # Example 3: Summary memory with custom prompt
    print("\nExample 3: Summary memory with custom prompt")
    conv_summary_custom = create_conversation_chain_with_custom_memory(
        "summary",
        prompt_template="Summarize the conversation so far: {chat_history}"
    )
    conv_summary_custom.predict(input="I like to read books.")
    conv_summary_custom.predict(input="My favorite genre is science fiction.")
    conv_summary_custom.predict(input="What do you know about me?")

def demonstrate_simple_sequential_chain():
    """Demonstrate SimpleSequentialChain usage"""
    print("\n=== SimpleSequentialChain Examples ===")
    
    llm = get_llm()
    
    # Example 1: Two-step process - Translate then summarize
    print("\nExample 1: Translate then Summarize")
    
    # First chain: Translate to Spanish
    translate_template = PromptTemplate(
        input_variables=["text"],
        template="Translate the following text to Spanish: {text}"
    )
    translate_chain = LLMChain(llm=llm, prompt=translate_template)
    
    # Second chain: Summarize the translation
    summarize_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one sentence: {text}"
    )
    summarize_chain = LLMChain(llm=llm, prompt=summarize_template)
    
    # Create sequential chain
    sequential_chain = SimpleSequentialChain(
        chains=[translate_chain, summarize_chain],
        verbose=True
    )
    
    # Test the chain
    result = sequential_chain.run("Python is a programming language used for web development, data science, and artificial intelligence.")
    print(f"Result: {result}")
    
    # Example 2: Three-step process - Analyze, improve, format
    print("\nExample 2: Analyze, Improve, Format")
    
    # First chain: Analyze the text
    analyze_template = PromptTemplate(
        input_variables=["text"],
        template="Analyze the following text and identify its main topic: {text}"
    )
    analyze_chain = LLMChain(llm=llm, prompt=analyze_template)
    
    # Second chain: Improve the analysis
    improve_template = PromptTemplate(
        input_variables=["text"],
        template="Take this analysis and make it more detailed and professional: {text}"
    )
    improve_chain = LLMChain(llm=llm, prompt=improve_template)
    
    # Third chain: Format the result
    format_template = PromptTemplate(
        input_variables=["text"],
        template="Format this text as a structured report with bullet points: {text}"
    )
    format_chain = LLMChain(llm=llm, prompt=format_template)
    
    # Create sequential chain
    three_step_chain = SimpleSequentialChain(
        chains=[analyze_chain, improve_chain, format_chain],
        verbose=True
    )
    
    # Test the chain
    result2 = three_step_chain.run("Machine learning is transforming how we approach problem-solving in technology.")
    print(f"Result: {result2}")
    
    # Example 3: Code generation and explanation
    print("\nExample 3: Code Generation and Explanation")
    
    # First chain: Generate code
    code_template = PromptTemplate(
        input_variables=["task"],
        template="Write a Python function to {task}. Return only the code, no explanations."
    )
    code_chain = LLMChain(llm=llm, prompt=code_template)
    
    # Second chain: Explain the code
    explain_template = PromptTemplate(
        input_variables=["text"],
        template="Explain this Python code in simple terms: {text}"
    )
    explain_chain = LLMChain(llm=llm, prompt=explain_template)
    
    # Create sequential chain
    code_explain_chain = SimpleSequentialChain(
        chains=[code_chain, explain_chain],
        verbose=True
    )
    
    # Test the chain
    result3 = code_explain_chain.run("calculate the factorial of a number")
    print(f"Result: {result3}")

def create_simple_sequential_chain_examples():
    """Create reusable SimpleSequentialChain examples"""
    
    llm = get_llm()
    
    # Example 1: Text processing pipeline
    def create_text_processing_chain():
        """Create a chain that translates then summarizes"""
        translate_template = PromptTemplate(
            input_variables=["text"],
            template="Translate to Spanish: {text}"
        )
        summarize_template = PromptTemplate(
            input_variables=["text"],
            template="Summarize in one sentence: {text}"
        )
        
        translate_chain = LLMChain(llm=llm, prompt=translate_template)
        summarize_chain = LLMChain(llm=llm, prompt=summarize_template)
        
        return SimpleSequentialChain(
            chains=[translate_chain, summarize_chain],
            verbose=False
        )
    
    # Example 2: Code generation pipeline
    def create_code_generation_chain():
        """Create a chain that generates code then explains it"""
        code_template = PromptTemplate(
            input_variables=["task"],
            template="Write Python code for: {task}"
        )
        explain_template = PromptTemplate(
            input_variables=["text"],
            template="Explain this code: {text}"
        )
        
        code_chain = LLMChain(llm=llm, prompt=code_template)
        explain_chain = LLMChain(llm=llm, prompt=explain_template)
        
        return SimpleSequentialChain(
            chains=[code_chain, explain_chain],
            verbose=False
        )
    
    # Example 3: Content creation pipeline
    def create_content_creation_chain():
        """Create a chain that generates content then formats it"""
        generate_template = PromptTemplate(
            input_variables=["topic"],
            template="Write a short article about: {topic}"
        )
        format_template = PromptTemplate(
            input_variables=["text"],
            template="Format this as a professional blog post with headings: {text}"
        )
        
        generate_chain = LLMChain(llm=llm, prompt=generate_template)
        format_chain = LLMChain(llm=llm, prompt=format_template)
        
        return SimpleSequentialChain(
            chains=[generate_chain, format_chain],
            verbose=False
        )
    
    return {
        "text_processing": create_text_processing_chain(),
        "code_generation": create_code_generation_chain(),
        "content_creation": create_content_creation_chain()
    }

def main():
    """Main function to run the tests"""
    try:
        test_conversation()
        test_with_system_instruction()
        test_different_memory_types()
        demonstrate_memory_parameters()
        demonstrate_simple_sequential_chain()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Azure deployment name and credentials.")

if __name__ == "__main__":
    main()

# Example of how to use these objects in other files:
"""
# In another file (e.g., my_app.py), you can import and use:

from test_memory import (
    get_llm, 
    get_conversation_chain, 
    get_memory, 
    MODEL_CONFIG,
    create_memory,
    create_conversation_chain_with_custom_memory,
    create_simple_sequential_chain_examples,
    demonstrate_simple_sequential_chain
)

# ===== Basic Usage =====
# Get the configured LLM
llm = get_llm()

# Get the conversation chain with default memory
conv = get_conversation_chain()

# Get just the memory
memory = get_memory()

# ===== Custom Memory Usage =====
# 1. Window Memory (stores last k messages)
conv_window = create_conversation_chain_with_custom_memory("window", k=5)

# 2. Token Buffer Memory (token-based limit)
conv_token = create_conversation_chain_with_custom_memory("token_buffer", max_token_limit=1000)

# 3. Summary Memory (summarizes old messages)
conv_summary = create_conversation_chain_with_custom_memory("summary")

# 4. Summary Buffer Memory (summary + recent buffer)
conv_summary_buffer = create_conversation_chain_with_custom_memory(
    "summary_buffer", 
    max_token_limit=500
)

# ===== Usage Examples =====
# Use the LLM directly
response = llm.invoke("Hello, how are you?")

# Use conversation chain with default memory
response = conv.predict(input="Tell me about Python")

# Use conversation chain with window memory
response = conv_window.predict(input="What did we talk about?")

# Access model configuration
print(f"Using model: {MODEL_CONFIG['model']}")
print(f"Temperature: {MODEL_CONFIG['temperature']}")

# Example with system instruction
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
response = llm.invoke(messages)

# ===== Memory Parameter Examples =====
# Create memory with custom parameters
memory_window = create_memory("window", k=3)
memory_token = create_memory("token_buffer", max_token_limit=200)
memory_summary = create_memory("summary", prompt_template="Summarize: {chat_history}")

# Create conversation chain with custom memory
conv_custom = create_conversation_chain_with_custom_memory(
    "window", 
    k=10, 
    return_messages=True
)

# ===== SimpleSequentialChain Examples =====
# Get pre-built sequential chain examples
chain_examples = create_simple_sequential_chain_examples()

# Use text processing chain (translate then summarize)
text_result = chain_examples["text_processing"].run("Python is a great programming language")

# Use code generation chain (generate code then explain)
code_result = chain_examples["code_generation"].run("calculate fibonacci numbers")

# Use content creation chain (generate content then format)
content_result = chain_examples["content_creation"].run("artificial intelligence")

# Create custom sequential chain
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate

llm = get_llm()

# Step 1: Generate ideas
ideas_template = PromptTemplate(
    input_variables=["topic"],
    template="Generate 3 creative ideas about: {topic}"
)
ideas_chain = LLMChain(llm=llm, prompt=ideas_template)

# Step 2: Evaluate ideas
evaluate_template = PromptTemplate(
    input_variables=["text"],
    template="Evaluate these ideas and pick the best one: {text}"
)
evaluate_chain = LLMChain(llm=llm, prompt=evaluate_template)

# Create the sequential chain
custom_chain = SimpleSequentialChain(
    chains=[ideas_chain, evaluate_chain],
    verbose=True
)

# Use the chain
result = custom_chain.run("sustainable energy solutions")
""" 