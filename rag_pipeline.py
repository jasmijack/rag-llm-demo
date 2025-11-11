"""
This script:
- Loads PDF documents from a folder
- Splits them into chunks
- Builds a Chroma vector store with OpenAI embeddings
- Uses a ChatOpenAI model to answer questions grounded in those documents
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def build_retriever(
    pdf_dir: str = "data/pdfs",
    collection_name: str = "ai_initiatives",
):
    """Create a retriever from all PDFs in pdf_dir."""
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    # 1. Load all PDFs
    loader = PyPDFDirectoryLoader(path=pdf_dir)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # 4. Vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
    )

    # 5. Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    return retriever


def create_llm():
    """Create the ChatOpenAI client used for answering questions."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=1.2,
        stop=["INST"],
    )


QNA_SYSTEM_MESSAGE = """
You are an assistant trained to answer questions using the supplied documents.
Base your reply only on the given context.
If the context does not mention the requested information, clearly respond that you do not have enough data.
Keep responses factual, concise, and free of assumptions.
"""

QNA_USER_MESSAGE_TEMPLATE = """
### Context
Here are some documents that are relevant to the question mentioned below.
{context}

### Question
{question}
"""


def build_prompt(context: str, question: str) -> str:
    """Format the chat prompt in the same pattern used in the notebook."""
    return (
        f"[INST]{QNA_SYSTEM_MESSAGE}\n\n"
        f"user: {QNA_USER_MESSAGE_TEMPLATE.format(context=context, question=question)}\n"
        "[/INST]"
    )


def rag_answer(question: str, retriever, llm) -> str:
    """Run a full RAG cycle and return the answer text."""
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "I could not find any relevant information in the documents."

    context_list = [d.page_content for d in docs]
    context = ". ".join(context_list)

    prompt = build_prompt(context=context, question=question)

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Sorry. I encountered the following error: {e}"


if __name__ == "__main__":
    pdf_dir = os.getenv("RAG_PDF_DIR", "data/pdfs")
    print(f"Loading documents from: {pdf_dir}")
    retriever = build_retriever(pdf_dir=pdf_dir)
    llm = create_llm()

    print("RAG demo ready. Type a question about your PDFs. Type 'exit' to quit.")
    while True:
        user_q = input("\nQuestion: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break
        answer = rag_answer(user_q, retriever, llm)
        print(f"\nAnswer:\n{answer}")
