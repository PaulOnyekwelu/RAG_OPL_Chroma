import time
import pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from .commons import load_document, chunk_data, calculate_embedding_cost, ask_and_get_answer


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


def init_rag_opl(index_name, load_type="file", file_path=None, query=None):
    document = (
        load_document(file_path) if load_type == "file" else load_from_wikipedia(query)
    )

    chunks = chunk_data(document)

    calculate_embedding_cost(chunks)
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
            break

        answer = ask_and_get_answer(vector_store, q)
        print(f"\n Answer: \n")
        print(f"{answer} \n")
        print("-" * 70)
