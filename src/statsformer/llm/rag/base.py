from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum

from statsformer.llm.prompting import TaskDescription


@dataclass
class RAGDocument:
    content: str
    metadata:  dict | None = field(default=None)

    def format(self):
        if self.metadata:
            return f"METADATA: {self.metadata};\nCONTENT: {self.content}\n"
        else:
            return self.content


class RAGTypes(Enum):
    DISABLED = "disabled"
    OMIM = "OMIM"


@dataclass
class RAGConfig:
    rag_type: str = field(
        default=RAGTypes.DISABLED.value,
        metadata=dict(
            help="Type of RAG system to use",
            choices=[e.value for e in RAGTypes]
        )
    )
    rag_topk: int = field(
        default=5, metadata=dict(
            help="Number of documents to retrieve"
        ))
    retriever_prompt: str = field(default="prompts/rag/default_retriever.txt")
    small_rag: bool = field(
        default=False, metadata=dict(
            help="Retrieve k documents for all features, as opposed to k per feature."
        )
    )
    rag_persist_directory: str = field(
        default=None, metadata=dict(
            help="Location of saved vector database"
        )
    )

    @classmethod
    def disabled(cls) -> "RAGConfig":
        return cls(
            rag_type=RAGTypes.DISABLED.value
        )

    def instantiate_rag_system(self) -> "RAGSystem":
        if self.rag_type == RAGTypes.OMIM.value:
            from statsformer.llm.rag.omim import OmimRAGSystem
            return OmimRAGSystem(config=self)
        elif self.rag_type == RAGTypes.DISABLED.value:
            return None
        else:
            raise ValueError(f"Unsupported RAG type: {self.rag_type}")


class RAGSystem(ABC):
    def __init__(
        self,
        config: RAGConfig
    ):
        self.config = config
    
    @abstractmethod
    def retrieve_docs(
        self,
        batch_features: list[str],
        task_description: TaskDescription,
        k: int=None
    ) -> list[RAGDocument]:
        pass
