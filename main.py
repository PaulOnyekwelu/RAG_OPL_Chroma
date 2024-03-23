if __name__ == "__main__":
    import os
    import streamlit as st
    from dotenv import load_dotenv, find_dotenv

    # from RAG.rag_opl import init_rag_opl
    # from RAG.rag_chroma import init_rag_chroma
    from RAG.rag_chroma import load_embeddings_chroma
    from RAG.commons import (
        load_document,
        chunk_data,
        calculate_embedding_cost,
        ask_and_get_answer,
    )

    load_dotenv(find_dotenv(), override=True)

    def clear_history():
        if "history" in st.session_state:
            del st.session_state["history"]

    # init_rag_opl(
    #     "internation-students-uk",
    #     file_path="./attention_is_all_you_need.pdf",
    # )

    # init_rag_chroma(file_path="./rag_powered_by_google_search.pdf")

    st.image("./img.png")
    st.subheader("LLM Question Answering Application")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key: ", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader("Upload file: ", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input(
            "chunk size: ",
            min_value=100,
            max_value=2048,
            value=512,
            on_change=clear_history,
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap: ",
            min_value=0,
            max_value=10,
            value=0,
            on_change=clear_history,
        )
        k = st.number_input(
            "K value: ", min_value=1, max_value=20, value=5, on_change=clear_history
        )
        add_data = st.button("Add Data", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading, Chunking, and embedding file..."):
                bytes_data = uploaded_file.read()
                _, ext = os.path.splitext(uploaded_file.name)
                file_name = os.path.join("./", f"uploaded_file{ext}")

                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                document = load_document(file_name)
                chunks = chunk_data(document, chunk_size, chunk_overlap)
                st.write(f"Chunk Size: {chunk_size}. Number of Chunks: {len(chunks)}")

                tokens, embedding_cost = calculate_embedding_cost(chunks)

                st.write(
                    f"Total tokens: {tokens}. Embedding Cost: {embedding_cost} USD"
                )

                vector_store = load_embeddings_chroma(
                    chunks=chunks, create_embeddings=True
                )

                st.session_state.vs = vector_store
                st.success("File uploaded, chunked, and embedded successfully...")

    q = st.text_input("Input Prompt: ")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer: ", value=answer["answer"], height=50)

            st.divider()
            if "history" not in st.session_state:
                st.session_state.history = ""

            value = f"Q: {q}\nA: {answer.get('answer')}"
            st.session_state.history = (
                f"{value} \n {'-' * 100} \n {st.session_state.history}"
            )
            h = st.session_state.history
            st.text_area(label="Chat History", value=h, key="history", height=400)
