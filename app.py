from rag_core import process_document, ask
import logging

# 配置日志记录
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 定义日志记录函数
def log_query(query, result):
    """记录每次查询的相关信息"""
    if result["empty_retrieval"]:
        logging.info(f"Query: {query} - No relevant content found.")
    else:
        logging.info(f"Query: {query} - Answer: {result['answer']} - Hits: {len(result['hits'])}")

def main():
    document_path = input("请输入文档路径：").strip().strip('"').strip("'")
    index, paragraphs_with_metadata = process_document(document_path)

    print("\n文档已加载完成，可以开始提问。输入 exit 退出。\n")

    while True:
        query = input("请输入你的问题：").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        result = ask(query, index, paragraphs_with_metadata, top_k=3, dist_threshold=1.2)

        # 调用日志记录函数
        log_query(query, result)

        # 打印生成的答案
        print("\n=== ANSWER ===")
        print(result["answer"])

        # 如果有检索到命中段落，展示详细信息
        if result["hits"]:
            print("\n=== SOURCES ===")
            for hit in result["hits"]:
                print(f'chunk_id={hit["chunk_id"]}, source={hit["source"]}, distance={hit["distance"]:.4f}')
        else:
            print("没有找到相关内容。\n")
        print()

if __name__ == "__main__":
    main()