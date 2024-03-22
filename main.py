if __name__ == "__main__":
    from RAG.rag_opl import init_rag_opl

    init_rag_opl(
        "internation-students-uk",
        file_path="./international_students_working_in_the_uk_during_your_studies_2015-16_booklet.pdf",
    )
