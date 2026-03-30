import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv


# ENV ----------------
load_dotenv()

st.set_page_config(page_title="Healthcare Intake Agent", layout="wide")


# LOAD & PROCESS MEDICAL DATA---------------

@st.cache_resource
def load_vector_db():
    loader = TextLoader("medical_data.txt")  
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# SYSTEM PROMPT ------------------------

system_prompt = """
You are a Healthcare Intake Assistant.

RULES:
- DO NOT provide diagnosis
- DO NOT prescribe medication
- Be empathetic, calm, and professional
- Ask structured follow-up questions
- Summarize for doctor
- Use retrieved medical context for guidance

Your job:
1. Collect patient symptoms
2. Ask relevant follow-ups
3. Structure patient history
4. Generate summary for doctor

Always say: "This is not a medical diagnosis."
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + """

Context:
{context}

User Input:
{question}

Response:
"""
)


# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)


# SESSION STATE------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        "symptoms": [],
        "duration": "",
        "severity": "",
        "medical_history": "",
        "medications": ""
    }



# UI ------------------------------
st.title("🩺 Healthcare Intake Agent")
st.markdown("### AI-powered preliminary symptom intake (Not a diagnosis)")




for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])


user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    response = qa_chain.run(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

# PATIENT SUMMARY GENERATION -----------------------------

if st.button("Generate Doctor Summary"):
    history_text = "\n".join([f"{c['role']}: {c['content']}" for c in st.session_state.chat_history])

    summary_prompt = f"""
    Convert the following conversation into a structured medical summary:

    {history_text}

    Format:
    - Symptoms
    - Duration
    - Severity
    - Medical History
    - Medications
    - Additional Notes
    """

    summary = llm.predict(summary_prompt)

    st.subheader("📄 Doctor Summary")
    st.write(summary)


