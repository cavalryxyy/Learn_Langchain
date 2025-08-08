import os
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
# Assuming your get_llm and embeddings are in these files
from test_memory import get_llm 
from AzureConnection import embeddings

# --- 1. Setup and Data Loading ---

# Ensure you have an environment variable for your LLM API key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" 

# Load the documents from the CSV file
loader = CSVLoader(file_path='OutdoorClothingCatalog_1000.csv')
docs = loader.load()

# Initialize your LLM model
llm = get_llm()

# Create a vector store index from the documents
# This will create embeddings and store them in memory for fast retrieval
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])


# --- 2. QA Chain Setup ---

# Set up the retrieval-based question-answering chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
    verbose=False, # Set to True to see chain details
)


# --- 3. Generate Question-Answer Pairs ---

# Use a generation chain to create question-answer pairs from your documents
# This creates the "ground truth" data for evaluation
# We'll generate questions for the first 5 documents for this example
# Note: This can take a few minutes depending on the LLM and number of docs
print("Generating Q&A pairs from documents...")
qa_generation_chain = QAGenerateChain.from_llm(llm)

# Convert documents to the format expected by QAGenerateChain
doc_inputs = [{"doc": doc.page_content} for doc in docs[:5]]
examples = qa_generation_chain.apply(doc_inputs)

# The output of .apply is a list of dicts, we need to extract the qa pairs
examples = [item['qa_pairs'] for item in examples]
print(f"Generated {len(examples)} sets of Q&A pairs.\n")

# Debug: Print the structure of the first example
if examples:
    print(f"First example structure: {examples[0]}")
    print(f"Available keys: {list(examples[0].keys())}\n")


# --- 4. Get Predictions from Your QA Chain ---

# Now, run your QA chain to get the "predicted" answers for the generated questions
print("Getting predictions from the QA chain...")
predictions = []
for example in examples:
    # Try different possible key names for the question
    question = example.get("question") or example.get("query") or example.get("text")
    if question:
        response = qa.invoke({"query": question})
        predictions.append({"result": response["result"]})
    else:
        print(f"No question found in example: {example}")
        predictions.append({"result": "No question available"})
print("Predictions received.\n")


# --- 5. Evaluate the Predictions ---

# Set up the evaluation chain
eval_chain = QAEvalChain.from_llm(llm)

# Get the graded outputs
# The eval chain compares the "ground truth" answer from `examples`
# with the "predicted" answer from `predictions`
print("Evaluating the predictions...")
graded_outputs = eval_chain.evaluate(examples, predictions)
print("Evaluation complete.\n")


# --- 6. Display Results ---

# Loop through the examples and the graded outputs to print the results
for i, eg in enumerate(examples):
    # Get the corresponding prediction and grade
    prediction = predictions[i]
    grade = graded_outputs[i]

    print(f"--- Example {i+1} ---")
    # Try different possible key names for the question
    question = eg.get("question") or eg.get("query") or eg.get("text")
    answer = eg.get("answer") or eg.get("result")
    
    print(f"Question: {question}")
    print(f"Real Answer: {answer}")
    print(f"Predicted Answer: {prediction['result']}")
    print(f"Predicted Grade: {grade['results']}\n")