import os
import tiktoken
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WikipediaLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain


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
    elif ext.lower() == ".txt":
        loader = TextLoader(file)
    else:
        raise TypeError(f"unsupported file type: {ext}")

    data = loader.load()

    print(f"file '{file}' loaded successfully...")

    return data


def load_from_wikipedia(query, lang="en", load_max_docs=2):

    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    chunks = loader.load()

    return chunks


def chunk_data(data, chunk_size=256, chunk_overlap=0):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    data = text_splitter.split_documents(data)

    return data


def calculate_embedding_cost(text):

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in text])
    return total_tokens, round(total_tokens / 1000 * 0.0004, 6)

    # print(f"Total Tokens: {total_tokens}")
    # print(f"Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}")


def create_prompt_template():
    system_template = r"""
    use the following piece of context to answer the user's question.
    -----------------------
    context: ```{context}```
    """
    user_template = """
    question: ```{question}```
    chat history: ```{chat_history}```
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
        ]
    )


def ask_and_get_answer(vector_store, query, k=5):
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.9)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    # chain = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=retriever
    # )
    # answer = crc.invoke(query)

    # adding memory to the chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # adding a prompt template
    qa_prompt = create_prompt_template()

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    answer = crc.invoke(query)

    return answer
