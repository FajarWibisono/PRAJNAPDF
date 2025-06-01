import streamlit as st

# 1. set_page_config HARUS setelah import streamlit tapi sebelum import lainnya
st.set_page_config(
    page_title="PRAJNA DocQA",
    page_icon="🧠",
    layout="wide"
)

# 2. Kode untuk menghilangkan menu
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# 3. Mengatasi konflik PyTorch dengan Streamlit
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"

# 4. import dependencies lainnya
import tempfile
import logging
import numpy as np
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Konfigurasi logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

@st.cache_resource
def load_embeddings(use_multilingual=True):
    """Load embeddings model with caching"""
    if use_multilingual:
        # Model multilingual untuk akurasi yang lebih baik dengan berbagai bahasa
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    else:
        # Model yang lebih ringan untuk kecepatan
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

def initialize_llm():
    """Inisialisasi model LLM dengan konfigurasi yang dioptimalkan"""
    try:
        model_config = {
            "temperature": 0.27,  # Lebih rendah untuk jawaban yang lebih konsisten
            "max_tokens": 3600,  # Ditingkatkan untuk jawaban yang lebih lengkap
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

@st.cache_data
def process_pdf_file(file_content, file_name):
    """Cache hasil pemrosesan PDF untuk menghindari pemrosesan ulang"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
            os.unlink(temp_file.name)
            return documents
    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}")
        return []

def smart_chunk_sampling(chunks, max_chunks):
    """Strategi sampling cerdas untuk memilih chunk yang representatif"""
    total_chunks = len(chunks)

    if total_chunks <= max_chunks:
        return chunks

    # Strategi 1: Prioritaskan bagian awal, tengah, dan akhir dokumen
    first_third = max_chunks // 3
    middle_third = max_chunks // 3
    last_third = max_chunks - first_third - middle_third

    # Ambil chunk dari awal dokumen
    first_part = chunks[:first_third]

    # Ambil chunk dari tengah dokumen
    middle_start = (total_chunks // 2) - (middle_third // 2)
    middle_part = chunks[middle_start:middle_start + middle_third]

    # Ambil chunk dari akhir dokumen
    last_part = chunks[-last_third:]

    return first_part + middle_part + last_part

def process_documents(pdf_docs, chunk_size=800, chunk_overlap=100, max_chunks=None, use_multilingual=True):
    """Memproses dokumen PDF dengan strategi pembatasan chunk yang lebih cerdas"""
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("Mengekstrak teks dari PDF...")
        progress_bar.progress(10)

        # Menggunakan text splitter yang lebih efisien
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        documents = []
        for pdf in pdf_docs:
            doc_content = process_pdf_file(pdf.read(), pdf.name)
            documents.extend(doc_content)

        if not documents:
            raise ValueError("Tidak ada dokumen yang berhasil diproses")

        progress_text.text("Membagi teks menjadi chunk...")
        progress_bar.progress(30)
        chunks = text_splitter.split_documents(documents)
        total_chunks = len(chunks)

        # Strategi pembatasan chunk yang lebih cerdas
        if max_chunks and total_chunks > max_chunks:
            st.info(f"Dokumen menghasilkan {total_chunks} chunk. Menggunakan strategi sampling cerdas untuk mempertahankan kualitas.")
            chunks = smart_chunk_sampling(chunks, max_chunks)
            st.info(f"Menggunakan {len(chunks)} chunk yang diambil dari seluruh dokumen.")

        progress_text.text(f"Membuat embedding untuk {len(chunks)} chunk...")
        progress_bar.progress(50)

        # Gunakan model embedding sesuai pilihan
        embeddings = load_embeddings(use_multilingual)

        progress_text.text("Membangun indeks pencarian...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        progress_bar.progress(90)

        progress_text.text("Pemrosesan selesai!")
        progress_bar.progress(100)
        return vector_store

    except Exception as e:
        logging.error(f"Error in document processing: {e}")
        raise

def get_conversation_chain(vector_store, search_k=3):
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
                search_kwargs={"k": search_k}  # Jumlah dokumen yang diambil
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False  # Ubah ke True hanya untuk debugging
        )

        return conversation_chain

    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        raise

def main():
    st.title("🧠 PRAJNA DocQA")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>Selamat datang di PRAJNA DocQA!</strong> - Asisten Pribadi Anda untuk analisis dokumen PDF.
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

        processing_mode = st.radio(
            "Mode Pemrosesan:",
            ["Cepat (kualitas lebih rendah)", "Seimbang", "Kualitas Tinggi (lebih lambat)"]
        )

        # Opsi lanjutan
        advanced_options = st.expander("Opsi Lanjutan")
        with advanced_options:
            use_multilingual = st.checkbox(
                "Gunakan model multilingual",
                value=True,
                help="Model multilingual lebih akurat untuk berbagai bahasa tetapi lebih lambat"
            )

            max_chunks = st.number_input(
                "Jumlah Maksimum Chunk (0 = tidak dibatasi)",
                min_value=0,
                max_value=2000,
                value=0,
                help="Membatasi jumlah chunk dapat mempercepat pemrosesan tetapi berpotensi mengurangi akurasi"
            )

            if max_chunks == 0:
                max_chunks = None

            search_k = st.slider(
                "Jumlah dokumen yang diambil per pertanyaan",
                min_value=1,
                max_value=10,
                value=3,
                help="Nilai lebih tinggi meningkatkan akurasi tetapi bisa membuat jawaban lebih panjang"
            )

        if st.button("🔄 Proses Dokumen", use_container_width=True):
            if not pdf_docs:
                st.error("Silakan unggah dokumen terlebih dahulu!")
                return

            # Sesuaikan parameter berdasarkan mode
            if processing_mode == "Cepat (kualitas lebih rendah)":
                chunk_size = 1000
                chunk_overlap = 50
            elif processing_mode == "Seimbang":
                chunk_size = 800
                chunk_overlap = 100
            else:
                chunk_size = 450
                chunk_overlap = 150

            with st.spinner("📊 Memproses dokumen..."):
                try:
                    st.session_state.vector_store = process_documents(
                        pdf_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        max_chunks=max_chunks,
                        use_multilingual=use_multilingual
                    )
                    st.session_state.conversation = get_conversation_chain(
                        st.session_state.vector_store,
                        search_k=search_k
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
                    error_msg = f"❌ Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}"
                    st.error(error_msg)
                    logging.error(f"Error in chat response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer dengan disclaimer yang lebih menarik
    st.markdown(
        """
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='font-size: 12px; color: #6c757d; margin: 0;'>
        ⚠️ <strong>Disclaimer:</strong> AI-LLM dapat membuat kesalahan.
        Mohon verifikasi informasi penting sebelum mengambil keputusan.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
