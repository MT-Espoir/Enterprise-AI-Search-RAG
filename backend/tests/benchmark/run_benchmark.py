import sys, os
import json
import statistics
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

import chromadb
from app.ingestion.embedder.google_embedder import GoogleEmbedder
from app.vectorstore.operations import VectorStoreOps
from app.core.retrieval import Retriever
from tests.benchmark.metrics import calculate_hit_rate, calculate_mrr, calculate_recall_at_k, calculate_precision_at_k

# ── Setup ChromaDB (Sử dụng DB từ E2E test) ─────
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'e2e_chromadb'))
COLLECTION_NAME = "e2e_test"

def run_evaluation():
    print("🚀 Khởi chạy Benchmark Evaluation cho Retriever...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Không tìm thấy database tại {DB_PATH}. Vui lòng chạy test_e2e_pipeline.py trước để tạo DB.")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Collection '{COLLECTION_NAME}' không tồn tại. Vui lòng chạy test_e2e_pipeline.py trước.")
        sys.exit(1)

    class MockVectorStoreOps(VectorStoreOps):
        def __init__(self, collection):
            self.collection = collection

    vector_ops = MockVectorStoreOps(chroma_collection)
    
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        print("❌ Thiếu GOOGLE_API_KEY trong .env")
        sys.exit(1)

    embedder = GoogleEmbedder(api_key=API_KEY)
    
    # Chúng ta sẽ lấy Top 5 để có thể tính metric cho K=3, K=5
    retriever = Retriever(ops=vector_ops, embedder=embedder, top_k=5)

    dataset_path = os.path.join(os.path.dirname(__file__), "evaluation_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"✅ Đã load {len(dataset)} câu hỏi test.")
    
    results = {
        "hit_rate": [],
        "mrr": [],
        "recall@3": [],
        "precision@3": [],
        "recall@5": [],
        "precision@5": [],
    }

    for idx, item in enumerate(dataset):
        question = item["question"]
        expected = item["expected_sources"]
        print(f"\n[{idx+1}/{len(dataset)}] Q: {question}")
        
        # Gọi retriever
        retrieved_chunks = retriever.retrieve(question)
        
        # Tính toán metric
        h_rate = calculate_hit_rate(retrieved_chunks, expected)
        mrr = calculate_mrr(retrieved_chunks, expected)
        r3 = calculate_recall_at_k(retrieved_chunks, expected, 3)
        p3 = calculate_precision_at_k(retrieved_chunks, expected, 3)
        r5 = calculate_recall_at_k(retrieved_chunks, expected, 5)
        p5 = calculate_precision_at_k(retrieved_chunks, expected, 5)

        results["hit_rate"].append(h_rate)
        results["mrr"].append(mrr)
        results["recall@3"].append(r3)
        results["precision@3"].append(p3)
        results["recall@5"].append(r5)
        results["precision@5"].append(p5)
        
        print(f"  > Hit: {h_rate}, MRR: {mrr:.2f}, R@3: {r3:.2f}, P@3: {p3:.2f}")

    # Báo cáo tổng kết
    print("\n" + "="*40)
    print("📊 BÁO CÁO TỔNG KẾT RETRIEVAL METRICS")
    print("="*40)
    print(f"Tổng số câu hỏi: {len(dataset)}")
    print(f"Hit Rate       : {statistics.mean(results['hit_rate']):.2f}")
    print(f"MRR            : {statistics.mean(results['mrr']):.2f}")
    print(f"Recall@3       : {statistics.mean(results['recall@3']):.2f}")
    print(f"Precision@3    : {statistics.mean(results['precision@3']):.2f}")
    print(f"Recall@5       : {statistics.mean(results['recall@5']):.2f}")
    print(f"Precision@5    : {statistics.mean(results['precision@5']):.2f}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
