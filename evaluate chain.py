from test_memory import get_llm
from AzureConnection import embeddings 
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAGenerateChain

# Setup QA system
loader = CSVLoader(file_path='OutdoorClothingCatalog_1000.csv')
docs = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings
).from_loaders([loader])

llm = get_llm()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": ""}
)

# Basic QA test
test_query = "What sun protection clothing do you have?"
response = qa.invoke({"query": test_query})
print("Basic QA Response:", response["result"])

# Evaluation examples
example_gen_chain = QAGenerateChain.from_llm(llm)

# Example 1: Test with specific, targeted questions
# Goal: Test QA system with questions that should have clear answers
print("\n--- Example 1: Specific Questions ---")
specific_questions = [
    "What is the price of the Sun Shield Shirt?",
    "Which products have UPF 50+ protection?",
    "What features does the UV Blocker Hoodie have?",
    "Which shirts are good for outdoor activities?"
]

for question in specific_questions:
    print(f"\nQ: {question}")
    response = qa.invoke({"query": question})
    print(f"A: {response['result'][:200]}...")

# Example 2: Test product comparison questions
# Goal: Test QA system's ability to compare and recommend products
print("\n--- Example 2: Comparison Questions ---")
comparison_questions = [
    "What are the differences between Sun Shield Shirt and Mountain Shirt?",
    "Which products are best for hiking?",
    "What are the most expensive items in the catalog?"
]

for question in comparison_questions:
    print(f"\nQ: {question}")
    response = qa.invoke({"query": question})
    print(f"A: {response['result'][:200]}...")

# Example 3: Test category-specific questions
# Goal: Test QA system on specific product categories
print("\n--- Example 3: Category Questions ---")
category_questions = [
    "List all shirts with sun protection",
    "What hoodies are available?",
    "Which products are under $50?"
]

for question in category_questions:
    print(f"\nQ: {question}")
    response = qa.invoke({"query": question})
    print(f"A: {response['result'][:200]}...")

# Example 4: Working QAGenerateChain with Retrieval Testing
print("\n--- Example 4: QAGenerateChain + Retrieval Testing ---")

qa_generator = QAGenerateChain.from_llm(llm)

# Test with multiple documents
for doc in docs[:5]:
    qa_pair = qa_generator.apply_and_parse([{"doc": doc.page_content}])
    question = qa_pair[0].get("question", qa_pair[0].get("query", "No question"))
    response = qa.invoke({"query": question})
    print(f"QA pair: {qa_pair}")