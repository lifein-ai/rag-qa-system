import os
import streamlit as st

from rag_core import process_document, rag_answer

# 页面基础设置
st.set_page_config(
    page_title="RAG 智能问答 Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG 智能问答 Demo")
st.write("上传一个 txt 文档，然后输入问题，系统会基于文档内容回答。")

# 确保上传目录存在
UPLOAD_DIR = "Text"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化 session_state
if "document_path" not in st.session_state:
    st.session_state.document_path = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "index" not in st.session_state:
    st.session_state.index = None

if "paragraphs_with_metadata" not in st.session_state:
    st.session_state.paragraphs_with_metadata = None

# 上传文件
uploaded_file = st.file_uploader("上传 txt 文档", type=["txt"])

if uploaded_file is not None:
    document_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    # 只有新文件才重新处理
    if st.session_state.uploaded_filename != uploaded_file.name:
        with open(document_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.document_path = document_path
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.document_ready = False

        st.success(f"文件已上传：{uploaded_file.name}")

        with st.spinner("正在处理文档，请稍候..."):
            index, paragraphs_with_metadata = process_document(document_path)

        st.session_state.index = index
        st.session_state.paragraphs_with_metadata = paragraphs_with_metadata
        st.session_state.document_ready = True

        st.success("文档处理完成，可以开始提问。")

# 显示当前文档
if st.session_state.uploaded_filename:
    st.info(f"当前文档：{st.session_state.uploaded_filename}")

# 输入问题
question = st.text_input("请输入你的问题：")

# 提问按钮
if st.button("开始提问"):
    if not st.session_state.document_path:
        st.warning("请先上传一个 txt 文档。")
    elif not st.session_state.document_ready:
        st.warning("文档还没有处理完成，请稍后再试。")
    elif not question.strip():
        st.warning("请输入问题。")
    else:
        with st.spinner("正在生成答案，请稍候..."):
            result = rag_answer(
                query=question,
                index=st.session_state.index,
                paragraphs_with_metadata=st.session_state.paragraphs_with_metadata
            )

        st.subheader("回答结果")
        st.write(result["answer"])

        # 显示检索状态
        if result["empty_retrieval"]:
            st.info("这次没有检索到足够相关的文档片段。")
        else:
            st.subheader("检索到的相关片段")
            for i, hit in enumerate(result["hits"], 1):
                with st.expander(f"片段 {i} | chunk_id={hit['chunk_id']} | distance={hit['distance']:.4f}"):
                    st.write(hit["text"])
                    st.caption(f"来源：{hit['source']}")