import streamlit as st
# Konfigurasi tampilan halaman
st.set_page_config(
    page_title="PRAJNA PDF",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.css-1rs6os.edgvbvh3 {visibility: hidden;}
.css-1lsmgbg.egzxvld0 {visibility: hidden;}
.css-14xtw13.e8zbici0 {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import os
import logging

# Masukkan kode pengecekan API key di sini
try:
    api_key = st.secrets["GROQ_API_KEY"]
    st.write("API Key berhasil dibaca")
except Exception as e:
    st.error(f"Error membaca API key: {e}")

# Setup logging dengan format yang lebih detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom CSS untuk tampilan yang lebih baik
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f0fe;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_llm():
    """Inisialisasi model LLM dengan konfigurasi yang dioptimalkan"""
    try:
        model_config = {
            "temperature": 0.36,  # Lebih rendah untuk jawaban yang lebih konsisten
            "max_tokens": 3699,  # Ditingkatkan untuk jawaban yang lebih lengkap
            "top_p": 0.9,  # Parameter tambahan untuk kualitas output
            "presence_penalty": 0.1,  # Mendorong variasi dalam respons
            "frequency_penalty": 0.1  # Mengurangi pengulangan
        }
        
        llm = ChatGroq(
            api_key=st.secrets["GROQ_API_KEY"],
            model_name="gemma2-9b-it",
            **model_config
        )
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        raise

def process_documents(pdf_docs):
    """Memproses dokumen PDF dengan penanganan error yang lebih baik"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,  # Ukuran chunk yang dioptimalkan
            chunk_overlap=150,  # Overlap yang lebih besar untuk konteks yang lebih baik
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Separator yang lebih detail
        )
        
        documents = []
        for pdf in pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf.read())
                loader = PyPDFLoader(temp_file.name)
                documents.extend(loader.load())
            os.unlink(temp_file.name)
        
        chunks = text_splitter.split_documents(documents)
        
        # Menggunakan model embedding multibahasa yang lebih baik
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    
    except Exception as e:
        logging.error(f"Error in document processing: {e}")
        raise

def get_conversation_chain(vector_store):
    """Membuat rantai percakapan dengan konfigurasi yang dioptimalkan"""
    try:
        llm = initialize_llm()
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 3}  # Mengambil 3 dokumen teratas
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        return conversation_chain
    
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        raise

def main():
    st.title("🧠 PRAJNA PDF")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>Selamat datang di PRAJNAPDF!</strong> - Asisten Pribadi Anda untuk analisis dokumen PDF.
        Unggah dokumen Anda dan ajukan pertanyaan untuk mendapatkan wawasan yang mendalam. Pastikan Anda memiliki koneksi internet yang baik.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar dengan tampilan yang lebih baik
    with st.sidebar:
        st.header("📁 Unggah Dokumen")
        pdf_docs = st.file_uploader(
            "Pilih file PDF Anda",
            type=["pdf"],
            accept_multiple_files=True,
            help="Anda dapat mengunggah beberapa file PDF sekaligus"
        )
        
        if st.button("🔄 Proses Dokumen", use_container_width=True):
            if not pdf_docs:
                st.error("Silakan unggah dokumen terlebih dahulu!")
                return
                
            with st.spinner("📊 Memproses dokumen..."):
                try:
                    st.session_state.vector_store = process_documents(pdf_docs)
                    st.session_state.conversation = get_conversation_chain(
                        st.session_state.vector_store
                    )
                    st.success("✅ Dokumen berhasil diproses!")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
                    logging.error(f"Error in main processing: {e}")

    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilan chat yang lebih baik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input chat dengan penanganan error yang lebih baik
    if "conversation" not in st.session_state:
        st.info("ℹ️ Silakan unggah dan proses dokumen PDF untuk memulai percakapan.")
    else:
        if prompt := st.chat_input("💭 Ketik pertanyaan Anda di sini..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    modified_prompt = (
                        "Berikan jawaban selalu dalam bahasa Indonesia yang jelas dan terstruktur, "
                        "kecuali jika diminta dalam bahasa Inggris. Jawaban harus mencakup: "
                        f"{prompt}"
                    )
                    
                    with st.spinner("🤔 Sedang berpikir..."):
                        response = st.session_state.conversation.invoke({"question": modified_prompt})
                        st.markdown(response["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                
                except Exception as e:
                    st.error("❌ Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.")
                    logging.error(f"Error in chat response: {e}")

    # Footer dengan disclaimer yang lebih menarik
    st.markdown(
        """
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='font-size: 13px; color: #6c757d; margin: 0;'>
        ⚠️ <strong>Disclaimer:</strong> AI-LLM dapat membuat kesalahan. 
        Mohon verifikasi informasi penting sebelum mengambil keputusan.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
