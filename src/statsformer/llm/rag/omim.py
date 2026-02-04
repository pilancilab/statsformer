import os
from statsformer.llm.prompting import TaskDescription, fill_in_prompt_file
from statsformer.llm.rag.base import RAGConfig, RAGDocument, RAGSystem
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from threading import Lock


OMIM_PERSIST_DIRECTORY = "data/rag_system/omim_vectorstore"

VECTORSTORE = {}
VECTORSTORE_LOCK = Lock()


class OmimRAGSystem(RAGSystem):
    def __init__(
        self,
        config: RAGConfig,
    ):
        super().__init__(config)
        omim_directory = OMIM_PERSIST_DIRECTORY \
            if self.config.rag_persist_directory is None \
                else self.config.rag_persist_directory

        embeddings = OpenAIEmbeddings()
        omim_directory = os.path.realpath(omim_directory)
        if os.path.exists(omim_directory):
            with VECTORSTORE_LOCK:
                if omim_directory in VECTORSTORE:
                    self.vectorstore = VECTORSTORE[omim_directory]
                else:
                    self.vectorstore = Chroma(
                        persist_directory=omim_directory,
                        embedding_function=embeddings
                    )
                    VECTORSTORE[omim_directory] = self.vectorstore
        else:
            raise FileNotFoundError(
                f"Vector store not found at {omim_directory}. Ensure data is preprocessed and saved.")        

    def _get_docs(
        self,
        features: list[str],
        task: str,
        k: int
    ) -> list[Document]:
        retrieval_query = fill_in_prompt_file(
            self.config.retriever_prompt,
            dict(
                task=task,
                features=str(features) if len(features) > 1 else features[0]
            )
        )
        return self.vectorstore.similarity_search(
            query=retrieval_query,
            k=k
        )

    def retrieve_docs(
        self,
        batch_features: list[str],
        task_description: TaskDescription,
        k: int=None
    ) -> list[RAGDocument]:
        task = task_description.task
        if k is None:
            k = self.config.rag_topk
        if not self.config.small_rag:
            docs = sum(
                [self._get_docs([gene], task, k) \
                    for gene in batch_features],
                start=[]
            )
        else:
            docs = self._get_docs(batch_features, task, k)
        
        docs = get_unique_docs(docs)
        return [
            RAGDocument(doc.page_content, doc.metadata) \
                for doc in docs
        ]


def get_unique_docs(docs: list[Document]) -> list[Document]:
    """
    Filters unique documents from a list of Document objects.

    Args:
        docs (list): List of Document objects.

    Returns:
        list: A list of unique Document objects.
    """
    seen = set()
    unique_docs = []

    for doc in docs:
        doc_id = (doc.page_content, tuple(sorted(doc.metadata.items())))
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)

    return unique_docs