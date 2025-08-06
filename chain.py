import os
from AzureConnection import AzureConnection
from langchain_openai import AzureChatOpenAI
from langchain.chains import (
    ConversationChain, 
    SimpleSequentialChain, 
    LLMChain,
    RouterChain
)
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptRouter
from langchain.chains.router.llm_router import LLMRouterChain
from langchain.chains.router.prompt import RouterPromptTemplate
from langchain.memory import ConversationBufferMemory

# Import model configuration from test_memory.py
from test_memory import MODEL_CONFIG, get_llm

def setup_llm():
    """Set up Azure OpenAI LLM with centralized configuration"""
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

def demonstrate_router_chain():
    """Demonstrate Router Chain usage"""
    print("\n=== Router Chain Examples ===")
    
    llm = setup_llm()
    
    # Define different prompt templates for different tasks
    python_template = PromptTemplate(
        input_variables=["question"],
        template="You are a Python programming expert. Answer this question: {question}"
    )
    
    math_template = PromptTemplate(
        input_variables=["question"],
        template="You are a mathematics expert. Solve this problem: {question}"
    )
    
    general_template = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer this question: {question}"
    )
    
    # Create the router chain
    router_prompt = RouterPromptTemplate.from_template(
        template="Given the following question, determine which expert should answer it:\n"
                "Question: {question}\n"
                "Experts:\n"
                "- Python Expert: For programming, coding, Python, software development questions\n"
                "- Math Expert: For mathematics, calculations, equations, statistics questions\n"
                "- General Expert: For all other questions\n"
                "Choose the most appropriate expert:"
    )
    
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    
    # Create destination chains
    python_chain = LLMChain(llm=llm, prompt=python_template)
    math_chain = LLMChain(llm=llm, prompt=math_template)
    general_chain = LLMChain(llm=llm, prompt=general_template)
    
    # Create the multi-prompt router
    router = MultiPromptRouter(
        router_chain=router_chain,
        destination_chains={
            "python": python_chain,
            "math": math_chain,
            "general": general_chain
        },
        default_chain=general_chain,
        verbose=True
    )
    
    # Test the router chain
    print("\nExample 1: Python Question")
    result1 = router.run("How do I create a list comprehension in Python?")
    print(f"Result: {result1}")
    
    print("\nExample 2: Math Question")
    result2 = router.run("What is the derivative of x^2 + 3x + 1?")
    print(f"Result: {result2}")
    
    print("\nExample 3: General Question")
    result3 = router.run("What is the capital of France?")
    print(f"Result: {result3}")
    
    return router

def demonstrate_sequential_chain():
    """Demonstrate SimpleSequentialChain usage"""
    print("\n=== SimpleSequentialChain Examples ===")
    
    llm = setup_llm()
    
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
    
    # Example 2: Code generation and explanation
    print("\nExample 2: Code Generation and Explanation")
    
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
    result2 = code_explain_chain.run("calculate the factorial of a number")
    print(f"Result: {result2}")

def create_router_chain_examples():
    """Create reusable Router Chain examples"""
    
    llm = setup_llm()
    
    # Example 1: Content type router
    def create_content_router():
        """Create a router for different content types"""
        
        # Define prompts for different content types
        technical_template = PromptTemplate(
            input_variables=["question"],
            template="You are a technical expert. Provide a detailed technical answer: {question}"
        )
        
        simple_template = PromptTemplate(
            input_variables=["question"],
            template="You are a teacher. Explain this in simple terms: {question}"
        )
        
        creative_template = PromptTemplate(
            input_variables=["question"],
            template="You are a creative writer. Provide an imaginative response: {question}"
        )
        
        # Router prompt
        router_prompt = RouterPromptTemplate.from_template(
            template="Determine the best approach for this question:\n"
                    "Question: {question}\n"
                    "Approaches:\n"
                    "- Technical: For complex, detailed technical explanations\n"
                    "- Simple: For basic explanations in simple terms\n"
                    "- Creative: For imaginative, creative responses\n"
                    "Choose the best approach:"
        )
        
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        
        # Create destination chains
        technical_chain = LLMChain(llm=llm, prompt=technical_template)
        simple_chain = LLMChain(llm=llm, prompt=simple_template)
        creative_chain = LLMChain(llm=llm, prompt=creative_template)
        
        return MultiPromptRouter(
            router_chain=router_chain,
            destination_chains={
                "technical": technical_chain,
                "simple": simple_chain,
                "creative": creative_chain
            },
            default_chain=simple_chain,
            verbose=False
        )
    
    return {
        "content_router": create_content_router()
    }

def create_sequential_chain_examples():
    """Create reusable SimpleSequentialChain examples"""
    
    llm = setup_llm()
    
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
    
    return {
        "text_processing": create_text_processing_chain(),
        "code_generation": create_code_generation_chain()
    }

def main():
    """Main function to run the chain examples"""
    try:
        print("=== Chain Examples ===")
        
        # Test Router Chain
        router = demonstrate_router_chain()
        
        # Test Sequential Chain
        demonstrate_sequential_chain()
        
        print("\n=== All chain examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Azure deployment name and credentials.")

if __name__ == "__main__":
    main() 