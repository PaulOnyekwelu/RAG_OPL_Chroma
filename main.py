if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    from RAG.rag_opl import init_rag_opl
    from RAG.rag_chroma import init_rag_chroma

    load_dotenv(find_dotenv(), override=True)

    # init_rag_opl(
    #     "internation-students-uk",
    #     file_path="./attention_is_all_you_need.pdf",
    # )

    init_rag_chroma(file_path="./rag_powered_by_google_search.pdf")
