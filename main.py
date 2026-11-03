import os
import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document as WordDocument
from pptx import Presentation
from bs4 import BeautifulSoup

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ==================================================
# STEP 1: CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

COLLECTION_NAME = "document_qa"


# ==================================================
# STEP 2: EXTRACT TEXT FROM DOCUMENTS
# ==================================================

def extract_text(file_path):

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    extracted_text = ""

    # PDF
    if extension == ".pdf":

        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(reader.pages):

            page_text = page.extract_text() or ""

            extracted_text += (
                f"\nPage {page_number + 1}\n"
                + page_text
            )

    # WORD DOCUMENT
    elif extension == ".docx":

        document = WordDocument(str(file_path))

        for paragraph in document.paragraphs:

            extracted_text += paragraph.text + "\n"

        for table in document.tables:

            for row in table.rows:

                extracted_text += " | ".join(
                    cell.text for cell in row.cells
                ) + "\n"

    # CSV
    elif extension == ".csv":

        dataframe = pd.read_csv(file_path)

        extracted_text = dataframe.to_string(index=False)

    # TSV
    elif extension == ".tsv":

        dataframe = pd.read_csv(
            file_path,
            sep="\t"
        )

        extracted_text = dataframe.to_string(index=False)

    # EXCEL
    elif extension in [".xlsx", ".xls"]:

        sheets = pd.read_excel(
            file_path,
            sheet_name=None
        )

        for sheet_name, dataframe in sheets.items():

            extracted_text += f"\nSheet: {sheet_name}\n"

            extracted_text += dataframe.to_string(
                index=False
            )

    # POWERPOINT
    elif extension == ".pptx":

        presentation = Presentation(str(file_path))

        for slide_number, slide in enumerate(
            presentation.slides
        ):

            extracted_text += (
                f"\nSlide {slide_number + 1}\n"
            )

            for shape in slide.shapes:

                if shape.has_text_frame:

                    extracted_text += shape.text + "\n"

                if shape.has_table:

                    for row in shape.table.rows:

                        extracted_text += " | ".join(
                            cell.text for cell in row.cells
                        ) + "\n"

    # JSON
    elif extension == ".json":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            extracted_text = json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )

    # HTML
    elif extension in [".html", ".htm"]:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            soup = BeautifulSoup(
                file.read(),
                "html.parser"
            )

            extracted_text = soup.get_text(
                separator="\n"
            )

    # XML
    elif extension == ".xml":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            soup = BeautifulSoup(
                file.read(),
                "xml"
            )

            extracted_text = soup.get_text(
                separator="\n"
            )

    # TEXT AND SOURCE CODE FILES
    else:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            extracted_text = file.read()

    return extracted_text.strip()


# ==================================================
# STEP 3: RAG ENGINE
# ==================================================

class RAGEngine:

    def __init__(self):

        if not GROQ_API_KEY:

            raise ValueError(
                "GROQ_API_KEY not found. "
                "Please check your .env file."
            )

        print("\nSTEP 1: Loading local embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print("Embedding model loaded successfully.")

        print("\nSTEP 2: Connecting to Groq...")

        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.2,
            max_retries=3,
            timeout=60,
            max_tokens=1200,
            reasoning_format="hidden"
        )

        print("Groq model configured successfully.")

        self.vector_store = None
        self.retriever = None

        self.chat_history = []

        # LangChain prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a document question-answering assistant.

                    Answer the user's question using the
                    information in the retrieved document context.

                    Rules:

                    1. Give clear and direct answers.
                    2. Do not invent information.
                    3. If the answer is not available in the
                       document context, say:
                       "I could not find that information
                       in the uploaded documents."
                    4. Use previous conversation only to
                       understand follow-up questions.
                    5. Use the retrieved document context
                       as the source of factual answers.
                    6. Do not provide unrelated information.

                    Previous conversation:
                    {chat_history}

                    Retrieved document context:
                    {context}
                    """
                ),
                (
                    "human",
                    "{question}"
                )
            ]
        )

        # LangChain chain:
        # Prompt -> Groq LLM -> String output
        self.rag_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("\nRAG engine is ready!")

    # ==================================================
    # STEP 4: CHUNK DOCUMENTS AND STORE IN CHROMADB
    # ==================================================

    def add_documents(self, documents):

        if not documents:

            print("No documents available.")

            return False

        print("\nSTEP 3: Splitting documents into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(documents)

        if not chunks:

            print("No chunks were created.")

            return False

        print(f"Total chunks created: {len(chunks)}")

        print("\nSTEP 4: Creating document embeddings...")

        # In-memory ChromaDB:
        # No old database deletion or Windows file-lock issue.
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME
        )

        print("Embeddings stored in ChromaDB.")

        # Create LangChain retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

        print("Document retriever created.")

        return True

    # ==================================================
    # STEP 5: RETRIEVE RELEVANT DOCUMENT CHUNKS
    # ==================================================

    def retrieve_documents(self, question):

        if self.retriever is None:

            raise ValueError(
                "Please upload documents first."
            )

        print("\nSTEP 5: Searching relevant document chunks...")

        relevant_documents = self.retriever.invoke(
            question
        )

        print(
            f"Retrieved {len(relevant_documents)} "
            "relevant chunks."
        )

        return relevant_documents

    # ==================================================
    # STEP 6: GENERATE ANSWER USING LANGCHAIN RAG
    # ==================================================

    def ask_question(self, question):

        # Retrieve relevant chunks from ChromaDB
        relevant_documents = self.retrieve_documents(
            question
        )

        if not relevant_documents:

            return (
                "I could not find relevant information "
                "in the uploaded documents.",
                []
            )

        # Combine retrieved chunks into context
        context = "\n\n".join(
            document.page_content
            for document in relevant_documents
        )

        # Prepare previous conversation
        history = ""

        for old_question, old_answer in self.chat_history[-5:]:

            history += (
                f"\nUser: {old_question}\n"
                f"Assistant: {old_answer}\n"
            )

        print("\nSTEP 6: Sending context and question to Groq...")

        # Execute LangChain RAG chain
        answer = self.rag_chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": history
            }
        )

        if not answer or not answer.strip():

            answer = (
                "The model returned an empty response. "
                "Please try again."
            )

        # Save conversation
        self.chat_history.append(
            (question, answer)
        )

        return answer, relevant_documents


# ==================================================
# STEP 7: UPLOAD DOCUMENTS
# ==================================================

def upload_documents(engine):

    print("\nUPLOAD YOUR DOCUMENTS")

    print("Enter the complete file paths.")
    print("Enter one file path per line.")
    print("Type 'done' when you finish.")

    file_paths = []

    while True:

        file_path = input("\nFile path: ").strip().strip('"')

        if file_path.lower() == "done":
            break

        if not file_path:
            continue

        if not Path(file_path).is_file():
            print("File not found. Please check the path.")
            continue

        file_paths.append(file_path)

    if not file_paths:
        print("\nNo documents selected.")
        return False

    documents = []

    print("\nSTEP 2: Extracting document text...")

    for file_path in file_paths:

        file_name = Path(file_path).name

        try:

            print(f"\nProcessing: {file_name}")

            text = extract_text(file_path)

            if not text:
                print("No readable text. Skipping.")
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "file_path": str(file_path)
                }
            )

            documents.append(document)

            print(f"Extracted {len(text)} characters.")

        except Exception as error:

            print(f"Error processing {file_name}: {error}")

    if not documents:
        print("\nNo documents could be processed.")
        return False

    return engine.add_documents(documents)


# ==================================================
# STEP 8: QUESTION AND ANSWER LOOP
# ==================================================

def question_loop(engine):

    print("\n" + "=" * 55)

    print("DOCUMENT QUESTION ANSWERING")

    print("=" * 55)

    print("\nYour documents are ready!")

    print("Ask questions related to your documents.")

    print("Type 'exit' to close the application.")

    while True:

        question = input(
            "\nYour question: "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("\nApplication closed.")

            break

        if not question:

            print("Please enter a question.")

            continue

        try:

            answer, sources = engine.ask_question(
                question
            )

            print("\nANSWER")
            print("-" * 55)

            print(answer)

            print("\nSOURCES")

            source_names = set(
                document.metadata.get(
                    "source",
                    "Unknown"
                )
                for document in sources
            )

            for source in source_names:

                print(f"- {source}")

            print("-" * 55)

        except Exception as error:

            print("\nUnable to generate answer.")

            print(f"Error: {error}")

            print(
                "\nCheck your Groq API key, model name "
                "and internet connection."
            )


# ==================================================
# STEP 9: MAIN FUNCTION
# ==================================================

def main():

    print("=" * 55)

    print("DOCUMENT QUESTION ANSWERING")

    print("LangChain + Groq + Hugging Face + ChromaDB")

    print("=" * 55)

    try:

        # Initialize RAG engine
        engine = RAGEngine()

        # Upload documents every time the program starts
        print("\nUPLOAD YOUR DOCUMENTS")

        uploaded = upload_documents(engine)

        if not uploaded:

            print("\nNo documents uploaded.")

            print("Application stopped.")

            return

        print("\nAll documents are ready!")

        # Start document question answering
        question_loop(engine)

    except Exception as error:

        print("\nAPPLICATION ERROR")

        print(error)


if __name__ == "__main__":

    main()
