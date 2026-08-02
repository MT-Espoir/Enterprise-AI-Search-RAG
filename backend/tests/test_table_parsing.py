import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.chunker.recursive_chunker import RecursiveChunker

def test_parsing():
    pdf_path = os.path.join(os.path.dirname(__file__), "test_table.pdf")
    
    # 1. Parse PDF
    parser = PDFParser()
    pages = parser.parse(pdf_path)
    
    print("=== KẾT QUẢ PARSE ===")
    for p in pages:
        print(f"Page {p.page_num}:")
        print(f"  - Text: {p.text}")
        print(f"  - Metadata tables: {len(p.metadata.get('tables', []))} tables found.")
        for idx, tbl in enumerate(p.metadata.get('tables', [])):
            print(f"    Table {idx+1}:\n{tbl}")

    # 2. Chunking
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_pages(pages)
    
    print("\n=== KẾT QUẢ CHUNK ===")
    for idx, chunk in enumerate(chunks):
        is_table = chunk["metadata"].get("is_table", False)
        print(f"Chunk {idx} (is_table={is_table}):")
        print(f"  Text: {repr(chunk['text'])}")

if __name__ == "__main__":
    test_parsing()
