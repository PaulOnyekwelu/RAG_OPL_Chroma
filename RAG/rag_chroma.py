import os
import time
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from .commons import load_document, chunk_data, print_embedding_cost, ask_and_get_answer

chroma_path = "./chroma_db"


def load_embeddings_chroma(
    persist_directory=chroma_path, chunks=None, create_embeddings=False
):
    dimension = 1536

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=dimension)
    if create_embeddings:
        print("Creating embeddings...")
        return Chroma.from_documents(
            chunks, embeddings, persist_directory=persist_directory
        )
    print("Embeddings already exist...")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def init_rag_chroma(
    chroma_path=chroma_path, load_type="file", file_path=None, query=None
):
    document = (
        load_document(file_path) if load_type == "file" else load_from_wikipedia(query)
    )

    chunks = chunk_data(document)

    print_embedding_cost(chunks)

    create_embeddings = not os.path.exists(chroma_path)
    vector_store = load_embeddings_chroma(
        persist_directory=chroma_path,
        chunks=chunks,
        create_embeddings=create_embeddings,
    )

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
