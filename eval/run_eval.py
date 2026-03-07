import os
import json
import csv
import sys

# 让 Python 能找到上一级目录的 rag_core.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag_core import process_document, ask


def load_eval_set(eval_file):
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    document_path = input("请输入评测文档路径：").strip().strip('"').strip("'")

    eval_file = os.path.join(os.path.dirname(__file__), "eval.jsonl")
    output_file = os.path.join(os.path.dirname(__file__), "eval_results.csv")

    eval_data = load_eval_set(eval_file)
    index, paragraphs = process_document(document_path)

    rows = []

    for item in eval_data:
        question = item["question"]
        gold_source = item.get("gold_source", "")
        answer_key_points = item.get("answer_key_points", [])

        result = ask(question, index, paragraphs, top_k=3, dist_threshold=1.2)

        hit_chunk_ids = [hit["chunk_id"] for hit in result["hits"]]
        hit_distances = [round(hit["distance"], 4) for hit in result["hits"]]

        rows.append({
            "question": question,
            "answer": result["answer"],
            "gold_source": gold_source,
            "answer_key_points": " | ".join(answer_key_points),
            "empty_retrieval": result["empty_retrieval"],
            "hit_chunk_ids": str(hit_chunk_ids),
            "hit_distances": str(hit_distances),
            "note": ""
        })

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "answer",
                "gold_source",
                "answer_key_points",
                "empty_retrieval",
                "hit_chunk_ids",
                "hit_distances",
                "note"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n评测完成，结果已保存到：{output_file}")


if __name__ == "__main__":
    main()