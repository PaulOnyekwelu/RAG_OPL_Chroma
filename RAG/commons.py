import os
import tiktoken
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WikipediaLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def load_document(file):
    """loads a document into an array of pages"""

    if file is None:
        raise ValueError("File is required")

    filename, ext = os.path.splitext(file)

    print(f"Loading file: '{file}'...")

    if ext.lower() == ".pdf":
        loader = PyPDFLoader(file)
    elif ext.lower() == ".docx":
        loader = Docx2txtLoader(file)
    else:
        raise TypeError(f"unsupported file type: {ext}")

    data = loader.load()

    print(f"file '{file}' loaded successfully...")

    return data


def load_from_wikipedia(query, lang="en", load_max_docs=2):

    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    chunks = loader.load()

    return chunks


def chunk_data(data, chunk_size=256):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    data = text_splitter.split_documents(data)

    return data


def print_embedding_cost(text):

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in text])
    print(f"Total Tokens: {total_tokens}")
    print(f"Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}")



def ask_and_get_answer(vector_store, query):
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.9)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.invoke(query)

    return answer