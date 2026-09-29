import streamlit as st
import requests
from urllib.parse import quote


# CONFIG

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# STYLE

st.markdown(
    """
    <style>

    .stApp {
        background-color: #161922;
        color: #F3F4F6;
    }

    [data-testid="stHeader"] {
        background-color: #161922;
    }

    [data-testid="stSidebar"] {
        background-color: #1C202A;
        border-right: 1px solid #303641;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #F8FAFC !important;
    }

    p {
        color: #C1C7D0;
    }

    div[data-testid="stMetric"] {
        background-color: #20252F;
        border: 1px solid #343B47;
        border-radius: 14px;
        padding: 1rem;
    }

    div[data-testid="stMetric"] label {
        color: #AAB2BF !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #20252F;
        border: 1px solid #343B47;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatInput"] {
        background-color: #20252F !important;
        border: 1px solid #3B4350 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #20252F !important;
        color: #F8FAFC !important;
        caret-color: white !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9CA3AF !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        color: #FFFFFF !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: #20252F;
        border-radius: 12px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background-color: #20252F;
        border: 1px dashed #4A5361;
        border-radius: 12px;
    }

    .stButton > button {
        background-color: #272D38;
        color: #F3F4F6;
        border: 1px solid #3B4350;
        border-radius: 9px;
    }

    .stButton > button:hover {
        border-color: #4F8CFF;
        color: #FFFFFF;
    }

    div[data-testid="stExpander"] {
        background-color: #20252F;
        border: 1px solid #343B47;
        border-radius: 10px;
    }

    div[role="radiogroup"] > label {
        padding: 0.6rem 0.7rem;
        border-radius: 9px;
        margin-bottom: 0.15rem;
    }

    div[role="radiogroup"] > label:hover {
        background-color: #272D38;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# SESSION

if "messages" not in st.session_state:
    st.session_state.messages = []


# API HELPERS

def get_api_health():

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )

        if response.status_code == 200:
            return True, response.json()

    except requests.RequestException:
        pass

    return False, {}


def get_documents():

    try:
        response = requests.get(
            f"{API_BASE_URL}/documents",
            timeout=5
        )

        if response.status_code == 200:
            return response.json().get(
                "documents",
                []
            )

    except requests.RequestException:
        pass

    return []


def get_error_message(response):

    try:
        return response.json().get(
            "detail",
            "Request failed."
        )

    except Exception:
        return (
            f"Request failed "
            f"({response.status_code})."
        )


# DATA

api_online, health_data = get_api_health()

documents = (
    get_documents()
    if api_online
    else []
)

document_count = health_data.get(
    "documents",
    0
)

chunk_count = health_data.get(
    "chunks",
    0
)


# SIDEBAR

with st.sidebar:

    st.markdown(
        "# 🧠✨ RAG Workspace"
    )

    st.caption(
        "Document Intelligence"
    )

    st.write("")

    st.caption(
        "WORKSPACE"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📚 Knowledge Base",
            "💬 Assistant",
            "⚙️ How it works"
        ],
        label_visibility="collapsed"
    )

    st.write("")

    st.caption(
        "SYSTEM"
    )

    if api_online:

        st.success(
            "● API online"
        )

    else:

        st.error(
            "● API offline"
        )

    st.write("")

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Docs",
            document_count
        )

    with metric2:

        st.metric(
            "Chunks",
            chunk_count
        )


# HOME

if page == "🏠 Home":

    st.title(
        "Welcome to RAG Workspace"
    )

    st.write(
        "Upload documents, build a searchable knowledge base "
        "and ask grounded questions using hybrid retrieval."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Documents indexed",
            document_count
        )

    with col2:

        st.metric(
            "🧱 Text chunks",
            chunk_count
        )

    with col3:

        st.metric(
            "🔎 Retrieval",
            "FAISS + BM25"
        )

    st.write("")

    st.subheader(
        "System status"
    )

    if api_online:

        st.success(
            "FastAPI backend is connected and ready."
        )

    else:

        st.error(
            "FastAPI backend is offline."
        )

        st.code(
            "python -m uvicorn api:app --reload",
            language="powershell"
        )

    st.write("")

    st.subheader(
        "Get started"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        with st.container(
            border=True,
            height=185
        ):

            st.markdown(
                "### 📤"
            )

            st.markdown(
                "**Upload documents**"
            )

            st.caption(
                "Add PDF files to your knowledge base."
            )

    with c2:

        with st.container(
            border=True,
            height=185
        ):

            st.markdown(
                "### 🧱"
            )

            st.markdown(
                "**Build your knowledge base**"
            )

            st.caption(
                "Documents are chunked and indexed automatically."
            )

    with c3:

        with st.container(
            border=True,
            height=185
        ):

            st.markdown(
                "### 💬"
            )

            st.markdown(
                "**Ask questions**"
            )

            st.caption(
                "Get grounded answers from your documents."
            )

    with c4:

        with st.container(
            border=True,
            height=185
        ):

            st.markdown(
                "### 📊"
            )

            st.markdown(
                "**Explore the pipeline**"
            )

            st.caption(
                "See how the RAG system works step by step."
            )


# KNOWLEDGE BASE

elif page == "📚 Knowledge Base":

    st.title(
        "Knowledge Base"
    )

    st.write(
        "Upload and manage the documents used by your RAG system."
    )

    st.write("")

    if not api_online:

        st.error(
            "FastAPI is offline. Start the backend first."
        )

    upload_col, stats_col = st.columns(
        [2, 1],
        gap="large"
    )

    with upload_col:

        st.subheader(
            "Upload knowledge"
        )

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"]
        )

        if uploaded_file is not None:

            if st.button(
                "📤 Add document",
                use_container_width=True,
                disabled=not api_online
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                try:

                    with st.spinner(
                        "Reading, chunking and indexing document..."
                    ):

                        response = requests.post(
                            f"{API_BASE_URL}/upload",
                            files=files,
                            timeout=600
                        )

                    if response.status_code == 200:

                        st.success(
                            f"{uploaded_file.name} indexed successfully."
                        )

                        st.rerun()

                    else:

                        st.error(
                            get_error_message(
                                response
                            )
                        )

                except requests.Timeout:

                    st.error(
                        "The upload took too long."
                    )

                except requests.RequestException as error:

                    st.error(
                        f"Could not connect to API: {error}"
                    )

    with stats_col:

        st.subheader(
            "Index"
        )

        st.metric(
            "Documents",
            document_count
        )

        st.metric(
            "Chunks",
            chunk_count
        )

    st.write("")

    st.subheader(
        "Indexed documents"
    )

    if not documents:

        st.info(
            "No documents indexed yet."
        )

    else:

        for document in documents:

            with st.container(
                border=True
            ):

                doc_col1, doc_col2 = st.columns(
                    [4, 1]
                )

                with doc_col1:

                    st.markdown(
                        f"### 📄 {document['name']}"
                    )

                    st.caption(
                        f"{document['chunks']} chunks"
                    )

                with doc_col2:

                    st.success(
                        "Indexed"
                    )

        st.write("")

        selected_document = st.selectbox(
            "Select a document to manage",
            [
                document["name"]
                for document in documents
            ]
        )

        if st.button(
            "🗑️ Remove selected document"
        ):

            safe_name = quote(
                selected_document,
                safe=""
            )

            try:

                response = requests.delete(
                    f"{API_BASE_URL}/documents/{safe_name}",
                    timeout=600
                )

                if response.status_code == 200:

                    st.success(
                        "Document removed."
                    )

                    st.rerun()

                else:

                    st.error(
                        get_error_message(
                            response
                        )
                    )

            except requests.RequestException as error:

                st.error(
                    f"Could not connect to API: {error}"
                )

        with st.expander(
            "⚠️ Danger zone"
        ):

            confirm_clear = st.checkbox(
                "I want to remove every indexed document."
            )

            if st.button(
                "Clear knowledge base",
                disabled=not confirm_clear
            ):

                try:

                    response = requests.delete(
                        f"{API_BASE_URL}/documents",
                        timeout=600
                    )

                    if response.status_code == 200:

                        st.session_state.messages = []

                        st.success(
                            "Knowledge base cleared."
                        )

                        st.rerun()

                    else:

                        st.error(
                            get_error_message(
                                response
                            )
                        )

                except requests.RequestException as error:

                    st.error(
                        f"Could not connect to API: {error}"
                    )


# ASSISTANT

elif page == "💬 Assistant":

    st.title(
        "AI Assistant"
    )

    st.write(
        "Ask questions about the documents stored in your knowledge base."
    )

    st.write("")

    if not api_online:

        st.error(
            "FastAPI is offline."
        )

        st.code(
            "python -m uvicorn api:app --reload",
            language="powershell"
        )

    elif document_count == 0:

        st.info(
            "Upload at least one PDF in the Knowledge Base first."
        )

    else:

        if not st.session_state.messages:

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    "Hello! I'm ready to answer questions "
                    "about your indexed documents."
                )

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )

                sources = message.get(
                    "sources",
                    []
                )

                if sources:

                    with st.expander(
                        f"📚 Sources ({len(sources)})"
                    ):

                        for source in sources:

                            with st.container(
                                border=True
                            ):

                                st.markdown(
                                    f"**📄 {source['document']}**"
                                )

                                st.caption(
                                    f"Page {source['page']}"
                                )

                                score1, score2, score3 = st.columns(
                                    3
                                )

                                with score1:

                                    st.metric(
                                        "Semantic",
                                        source.get(
                                            "semantic_score",
                                            0
                                        )
                                    )

                                with score2:

                                    st.metric(
                                        "BM25",
                                        source.get(
                                            "keyword_score",
                                            0
                                        )
                                    )

                                with score3:

                                    st.metric(
                                        "Hybrid",
                                        source.get(
                                            "hybrid_score",
                                            0
                                        )
                                    )

        question = st.chat_input(
            "Ask a question about your documents..."
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            try:

                with st.spinner(
                    "Searching your knowledge base..."
                ):

                    response = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={
                            "question": question
                        },
                        timeout=600
                    )

                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer returned."
                    )

                    sources = data.get(
                        "sources",
                        []
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                else:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "Backend error: "
                                + get_error_message(
                                    response
                                )
                            ),
                            "sources": []
                        }
                    )

            except requests.Timeout:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "The answer took too long to generate."
                        ),
                        "sources": []
                    }
                )

            except requests.RequestException as error:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Could not connect to API: {error}"
                        ),
                        "sources": []
                    }
                )

            st.rerun()

        if st.session_state.messages:

            st.write("")

            if st.button(
                "🗑️ Clear conversation"
            ):

                st.session_state.messages = []

                st.rerun()


# HOW IT WORKS

elif page == "⚙️ How it works":

    st.title(
        "How it works"
    )

    st.write(
        "The main steps behind the Retrieval-Augmented Generation pipeline."
    )

    st.write("")

    st.subheader(
        "Document processing"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(
            border=True,
            height=180
        ):

            st.markdown(
                "### 1. 📄 PDF Upload"
            )

            st.caption(
                "Documents are added to the knowledge base."
            )

    with c2:

        with st.container(
            border=True,
            height=180
        ):

            st.markdown(
                "### 2. ✂️ Chunking"
            )

            st.caption(
                "Text is split into smaller passages."
            )

    with c3:

        with st.container(
            border=True,
            height=180
        ):

            st.markdown(
                "### 3. 🧠 Embeddings"
            )

            st.caption(
                "Chunks are transformed into vector representations."
            )

    st.write("")

    st.subheader(
        "Retrieval"
    )

    c4, c5, c6 = st.columns(3)

    with c4:

        with st.container(
            border=True,
            height=180
        ):

            st.markdown(
                "### 4. 🔎 FAISS"
            )

            st.caption(
                "Semantic search finds passages similar to the question."
            )

    with c5:

        with st.container(
            border=True,
            height=180
        ):

            st.markdown(
                "### 5. 📚 BM25"
            )

            st.caption(
                "Lexical search finds passages with matching terms."
            )

    with c6:

        with st.container(
            border=True,
            height=180
        ):

            st.markdown(
                "### 6. ⚡ Hybrid Retrieval"
            )

            st.caption(
                "FAISS and BM25 results are combined."
            )

    st.write("")

    st.subheader(
        "Generation"
    )

    c7, c8 = st.columns(2)

    with c7:

        with st.container(
            border=True,
            height=175
        ):

            st.markdown(
                "### 7. 📦 Context"
            )

            st.caption(
                "The most relevant passages are selected "
                "and sent to the language model."
            )

    with c8:

        with st.container(
            border=True,
            height=175
        ):

            st.markdown(
                "### 8. 🤖 Qwen"
            )

            st.caption(
                "Qwen generates an answer using the retrieved context."
            )

    st.write("")

    st.subheader(
        "Architecture"
    )

    st.code(
        """
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
PDF Processing
        ↓
Chunking
        ↓
Embeddings
        ↓
FAISS + BM25
        ↓
Hybrid Retrieval
        ↓
Qwen
        ↓
Answer + Sources
        """,
        language="text"
    )