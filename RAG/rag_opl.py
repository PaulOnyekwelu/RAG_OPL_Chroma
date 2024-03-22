import os
import time
import tiktoken
import pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI




load_dotenv(find_dotenv(), override=True)


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


def insert_or_fetch_embeddings(index_name, chunks):
    dimension = 1536

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=dimension)

    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exist. Retrieving embeddings...", end="")
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print("ok")
    else:
        print(f"Creating index '{index_name}' and embeddings...", end="")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )
        print("ok")

    return vector_store


def delete_index(index_name="all"):
    pc = pinecone.Pinecone()

    if index_name == "all":
        print(f"deleting all indexes...", end="")
        index_names = pc.list_indexes().names()

        for idx in index_names:
            pc.delete_index(idx)

        print("ok")
    else:
        print("Deleting index '{index_name}'...", end="")
        pc.delete_index(index_name)
        print("ok")


def delete_other_indexes(index_name):
    pc = pinecone.Pinecone()
    indexes = pc.list_indexes().names()

    if len(indexes) > 1 or index_name not in indexes:
        delete_index()


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


def init_rag_opl(index_name, load_type="file", file_path=None, query=None):
    document = (
        load_document(file_path) if load_type == "file" else load_from_wikipedia(query)
    )

    chunks = chunk_data(document)

    print_embedding_cost(chunks)
    delete_other_indexes(index_name)

    vector_store = insert_or_fetch_embeddings(index_name, chunks)

    q_num = 1

    print("input 'exit' or 'quit' to exit")
    while True:
        q = input(f"Question {q_num}:  ")
        q_num += 1

        if q.lower() in ["quit", "exit"]:
            print("Quitting...", end="")
            time.sleep(2)
            print("goodbye!")

        answer = ask_and_get_answer(vector_store, q)
        print(f"\n Answer: \n")
        print(f"{answer} \n")
        print("-" * 70)