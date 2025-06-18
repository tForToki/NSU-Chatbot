import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch import cuda, bfloat16
import transformers

# Cache model and embeddings initialization
@st.cache_resource
def initialize():
    # Load documents
    loader = TextLoader("extracted_text.txt", encoding="utf8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    # Initialize embeddings and vectorstore
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="chroma_db")
    retriever = vectordb.as_retriever()

    # Load LLM
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token="Add_Your_Token_from_huggingface", #apply for a hugginface token for  "meta-llama/Llama-2-7b-chat-hf" and replace the `Add_Your_Token_from_huggingface` with the token
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token="Add_Your_Token_from_huggingface") #also replace here
    query_pipeline = transformers.pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=query_pipeline)

    return retriever, llm

retriever, llm = initialize()

# Prompt and Chain
RESPONSE_TEMPLATE = """[INST]
<>Context:
    {context}

INSTRUCTION:
    You are an AI Assistant for North South University (NSU), specialized in the Electrical and Computer Engineering (ECE) Department.
Your responses must strictly follow these guidelines:

1. **Use Contextual Information Only**:
   - Always base your responses on the provided context. Use exact phrases, facts, and data from the context without alteration.
   - If no relevant context is available, explicitly state: "I don’t have information on that topic."

2. **Focus on Relevance**:
   - Only respond to questions related to NSU or its ECE department.
   - For irrelevant questions, respond with: "This question is not related to North South University’s Electrical and Computer Engineering Department, so I cannot answer it."

3. **Avoid Assumptions**:
   - Do not generate or assume information beyond what is provided in the context.
   - Do not answer questions about other institutions, departments, or unrelated topics.

4. **Eliminate Redundancy**:
   - Avoid repeating the same information within a single response.
   - If the context contains duplicate details, consolidate the information into a single, coherent answer.

5. **Concise and Clear Responses**:
   - Keep answers direct, to the point, and focused on the question being asked.
   - Avoid unnecessary elaboration or irrelevant details.

6. **General NSU Questions**:
   - If a question about NSU is general (e.g., "Where is NSU located?"), provide a concise answer if the information is in the context.
   - If such general information is not in the context, respond with: "I don’t have that information in the provided context."

7. **Error Handling**:
   - If a query cannot be answered due to missing or irrelevant context, politely decline with a response like: "I don’t have enough information to answer that based on the provided context."

8. **Formatting and Clarity**:
   - Use a clear and structured format for your answers. Avoid overly verbose or technical language unless explicitly needed.

**Examples**:

1. **Question**: "What courses does the ECE department offer?"
   **Response**: "The ECE department at NSU offers courses such as [specific course names from context]."
   *(If the context provides exact course names, list them. If not, state: "I don’t have that information in the context.")*

2. **Question**: "What is the location of NSU?"
   **Response**: "I don’t have information about NSU’s location in the provided context."

3. **Question**: "What are the admission requirements for NSU's ECE department?"
   **Response**: "The admission requirements for the ECE department at NSU include [specific details from the context]."
   *(If not available in the context, respond: "I don’t have that information in the provided context.")*

Question: {question} [/INST]
Helpful Answer:
"""
PROMPT = PromptTemplate.from_template(RESPONSE_TEMPLATE)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={"verbose": False, "prompt": PROMPT},
)

# Streamlit App
st.title("NSU ECE Chatbot")
st.write("Ask any question related to North South University (NSU) Electrical and Computer Engineering (ECE) Department.")

query = st.text_input("Enter your question:", "")

if query:
    result = qa_chain.run(query) # Extracting the response from the result
    if isinstance(result, str) and "Helpful Answer" in result:
        cleaned_answer = result.split("Helpful Answer:", 1)[-1].strip()
    else:
        cleaned_answer = result
    
    # Display the response in the app
    st.subheader("Answer:")
    st.write(cleaned_answer)
