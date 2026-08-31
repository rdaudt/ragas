import streamlit as st

from ragas_demo.config import Settings
from ragas_demo.models import AnswerResult
from ragas_demo.runtime import create_rag_service

st.set_page_config(page_title="RAGAS Evaluation Demo", page_icon="📚")
st.title("RAGAS Evaluation Demo")
st.caption("Standalone questions grounded only in the seven indexed PDF documents.")


@st.cache_resource
def get_service():
    settings = Settings()
    return create_rag_service(settings)


def render_result(result: AnswerResult) -> None:
    st.markdown(result.answer)
    if result.sources:
        st.caption("Retrieved sources")
    for source in result.sources:
        label = f"{source.source_file} · page {source.page} · similarity {source.score:.3f}"
        with st.expander(label):
            st.text(source.text)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["question"])
    with st.chat_message("assistant"):
        render_result(message["result"])

if question := st.chat_input("Ask a question about the indexed documents"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = get_service().answer(question)
        except Exception as exc:
            st.error(f"Unable to answer: {exc}")
        else:
            render_result(result)
            st.session_state.messages.append({"question": question, "result": result})

