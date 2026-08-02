from .basechunker import BaseChunker


class SemanticChunker(BaseChunker):

    def chunk_text(self, text: str) -> list[str]:
        raise NotImplementedError(
            "Semantic chunking chưa được implement."
        )
