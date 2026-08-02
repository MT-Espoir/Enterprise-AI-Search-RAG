from .basechunker  import BaseChunker


class PropositionChunker(BaseChunker):

    def __init__(self, llm):
        self.llm = llm

    def chunk_text(self, text: str) -> list[str]:
        raise NotImplementedError(
            "Notebook Proposition Chunking sẽ implement sau."
        )