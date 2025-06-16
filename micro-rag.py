import sys
import os
import io
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QThread, pyqtSignal

# Suppress noisy logs from Google API client
logging.basicConfig(level=logging.WARNING)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# Install necessary packages if they are not already installed
try:
    from langchain_community.document_loaders.base import BaseLoader
    from langchain.docstore.document import Document
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:
    print("Some required libraries are not installed. Installing them now...")
    os.system(
        "pip install PyQt5 langchain langchain_community langchain-openai faiss-cpu sentence-transformers google-api-python-client google-auth-httplib2 google-auth-oauthlib pypdf")
    print("Installation complete. Please run the script again.")
    sys.exit()

# --- Configuration ---
DRIVE_FOLDER_ID = "1vmwYF8wIxb0FITaMf6ctfM3fyhWoeuSg"

# It's recommended to use environment variables for security
OPENAI_API_KEY = "sk-zA4OW76PeQCe4M4NISSuCW2sSAcxXiMjJvN7JPPpuGVWEkF7"
OPENAI_API_BASE = "https://api.chatanywhere.tech/v1"  # Or your custom base URL

# --- Part 1: Google Drive Loader Service ---

# Scopes define the level of access you are requesting.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveLoader(BaseLoader):
    """
    A LangChain compatible loader for Google Drive folders.
    It loads Google Docs and PDFs from a specified folder.
    """

    def __init__(self, folder_id: str):
        """
        Initializes the loader with a Google Drive folder ID.
        Args:
            folder_id: The ID of the Google Drive folder.
        """
        if not folder_id or folder_id == "YOUR_GOOGLE_DRIVE_FOLDER_ID":
            raise ValueError("Google Drive Folder ID is not set. Please edit the DRIVE_FOLDER_ID variable.")
        self.folder_id = folder_id
        self.creds = self._get_credentials()
        self.drive_service = build("drive", "v3", credentials=self.creds)

    def _get_credentials(self):
        """Handles Google authentication, including the OAuth 2.0 flow."""
        creds = None
        # The file token.json stores the user's access and refresh tokens.
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists("credentials.json"):
                    raise FileNotFoundError(
                        "Error: credentials.json not found. "
                        "Please follow the setup instructions to download it from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds

    def load(self):
        """
        Lists files in the folder, downloads them, extracts text,
        and returns a list of LangChain Document objects.
        """
        documents = []
        query = (
            f"'{self.folder_id}' in parents and "
            "(mimeType='application/vnd.google-apps.document' or "
            "mimeType='application/pdf') and "
            "trashed=false"
        )

        try:
            results = self.drive_service.files().list(
                q=query,
                pageSize=150,  # Adjust if you have more than 150 files
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
        except Exception as e:
            print(f"An error occurred while accessing the Drive API: {e}")
            print("Please ensure your DRIVE_FOLDER_ID is correct and you have access.")
            return []

        items = results.get("files", [])

        if not items:
            print(f"No Google Docs or PDFs found in the folder with ID: {self.folder_id}")
            return []

        print(f"Found {len(items)} files in Google Drive folder. Starting download and processing...")

        for item in items:
            file_id = item["id"]
            file_name = item["name"]
            mime_type = item["mimeType"]
            print(f"  - Processing '{file_name}' ({mime_type})...")

            try:
                if mime_type == 'application/vnd.google-apps.document':
                    # Export Google Doc as plain text
                    request = self.drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
                    content_bytes = request.execute()
                    content = content_bytes.decode('utf-8')
                elif mime_type == 'application/pdf':
                    # Download PDF and extract text
                    request = self.drive_service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                    fh.seek(0)
                    # Use pypdf to extract text from the PDF
                    from pypdf import PdfReader
                    reader = PdfReader(fh)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() or ""
                else:
                    continue

                metadata = {"source": f"{file_name} (Drive ID: {file_id})"}
                documents.append(Document(page_content=content, metadata=metadata))

            except Exception as e:
                print(f"    - Failed to process file '{file_name}': {e}")

        print("Finished processing all files.")
        return documents


# --- Part 2: RAG Pipeline Setup (in a background thread) ---

class RagSetupThread(QThread):
    """
    A QThread to handle the time-consuming setup of the RAG pipeline
    without freezing the GUI.
    """
    finished = pyqtSignal(object, str)  # Emits the QA chain object and a status message

    def run(self):
        """The main logic for setting up the RAG pipeline."""
        try:
            # 1. Load documents from Google Drive
            self.finished.emit(None, "Status: Connecting to Google Drive...")
            loader = GoogleDriveLoader(folder_id=DRIVE_FOLDER_ID)
            documents = loader.load()

            if not documents:
                self.finished.emit(None,
                                   f"Error: No documents found or could not access Drive folder. Please check your Folder ID and permissions.")
                return

            self.finished.emit(None, f"Status: Loaded {len(documents)} documents. Splitting text...")

            # 2. Create text splitter
            text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            docs_split = text_splitter.split_documents(documents)

            self.finished.emit(None, f"Status: Split into {len(docs_split)} chunks. Initializing embedding model...")

            # 3. Initialize embedding model
            model_name = "moka-ai/m3e-base"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为文本生成向量表示用于文本检索"
            )

            self.finished.emit(None, "Status: Creating FAISS vector database. This may take a while...")

            # 4. Create FAISS vector store and retriever
            db = FAISS.from_documents(docs_split, embedding)
            retriever = db.as_retriever()

            self.finished.emit(None, "Status: Initializing language model...")

            # 5. Initialize LLM
            if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
                self.finished.emit(None, "Error: OpenAI API key is not set. Please edit the OPENAI_API_KEY variable.")
                return

            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY
            )

            # 6. Build Conversational RAG chain with custom prompt
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

            # --- MODIFICATION START ---
            # This section implements the custom prompt logic as requested.

            # Template for generating the final answer based on context
            qa_template_str = """---

**你是一个牙膏制造业微生物管理知识解答人工助手，分析用户（输入）的当前需求，按照下列Prompt指令要求，经过反复推理和思考后回复用户。

（标准文档问答指令已根据我们现场的具体需求和问题进行了修改。每次有新的学习成果，都会对提示进行调整，以加入更好的指令，从而获得更优的回复。）

**在制定任何回复之前，必须仔细审查所提供的每一份文档，提取所有相关信息。确保回复符合相互独立且完全穷尽（MECE）原则，并以超链接方式引用所有相关文档。如出现潜在差异或需澄清之处，请重新审查文档以确保准确无误。**

**语气应正式且易于理解，始终提供准确的答案。**

**您的目标是只回答与已上传文件相关的问题。**

**您必须始终遵循以下指令：**

---

### 1. 全面审查文档

- 全面审查所有已上传文档，应用MECE原则，确保信息捕捉完整且无重叠。
- 不论材料中是否夹带英文或中文，请全部将回复翻译为中文。
- 细致地识别并提取相关信息。

### 2. 明确引用

- 在每次回复中都要明确引用每份文档的所有相关参考内容，确保信息来源可核查。

### 3. 聚焦具体问题

- 通过用户提问中的关键词，关注与问题相关的具体文档部分，确保不遗漏任何关键细节。包括所有协议和备忘录等相关文件。
#### 例子1: M01885的检测方法
#### 推理过程：M01885属于M开头产品，这种代号统一属于原料，应该找到原料微生物抽样及检测方法相关文件 TIB-029对应内容
#### 回答：QA-MICRO-TIB-029和原料检测具体内容 【Citation】
#### 例子2: 微生物实验室的参数要求
#### 推理过程：提取关键词，实验室，参数要求。根据要求匹配对应文件标题，应该是微生物洁净实验室管理程序 QA-MICRO-TIB-060
#### 回答：QA-MICRO-TIB-060和实验室参数要求具体内容  【Citation】


### 4. 结构化回复格式

- 以表格等结构化形式完整清晰展示信息。适用时请包含图表或流程图。

### 5. 完整性审核

- 对所有回复和重点信息进行交叉核查，确保全面性及所有相关信息的纳入。

### 6. 核查当前日期

- 在任何提示开头，务必使用datetime包识别当前日期。

### 7. 节假日问题

- 针对节假日问题，仅根据当前日期进行回答。

---

**对于与上传文件无关的问题，回复如下：**

```
未能在上传的文档中找到您的问题答案。
```

**不要描述您的答题过程，也不要要求额外的指示。**

---

## 标准推理过程

###第一阶段：思考与解答


1 .首先，你需要对用户问题进行拆解和分析，洞察用户的意图以及解决用户问题的最佳方式;
2,接看，你将使用该方式对用户问题进行一步步，带有思考过程的解答；
3 .使用〈解答〉"解答〉标签组织你的回答。


###第二阶段：反思


1 .在该阶段，你需要结合用户问题对用户的解答步骤和解答质量进行思考验证，认真审查你第一步给出的答案，从以下几个方面进行反思：
***内容的准确性:**用户的问题是否得到了完整和准确的回答？答案中是否存在任何事实性错误、逻辑问题或不一 致的地方？ 
***内容的完整性:**答案是否涵盖了所有关键信息？是否遗漏了任何重要的细节或方面？ 
***表达的清晰度:**答案的结构是否清晰易懂？语言表达是否流畅自然？是否使用了过于专业或难以理解的词汇？ 
***其他潜在问题:**除此之外，你认为答案还存在哪些问题或不足？
3 .使用〈反思〉〈反思〉标签组织你的回答，回到文件库中再次检索对应文件，提供Citation依据再回答。


##第三阶段：回答


1 .在该阶段，你应该根擦宗合前两阶段的内容为用户提供最终准确的回答;
2 .你的回答应该是结构清晰，逻辑严谨的；
3 .在回答过程中不要提及前两阶段的信息；


## 标准回复格式

---

**参考材料**：  
[请提供所有相关文件中对完整回复有帮助的原文引用，请勿遗漏任何文档。]

**参考文件**：  
[请提供所有查阅到的文档引用。]

**完整回复**：  
在扫描所有已上传文档后，请始终并且仅以表格形式，逐步完整地作答，表中应包括：

| 行动 | 责任人 | 源PDF文档链接 | 示例和详细说明（如有需要） | 相关章节 / 段落编号 | 备注 |
|------|--------|---------------|--------------------------|--------------------|------|

分享所有相关内容（注：需注明具体章节、段落及行号）

**警告**：  
AI助手虽智能，但仍可能犯错。员工有责任核查所提供答案的准确性和完整性。本助手仅供参考和指导用途，针对关键事项请在行动前咨询人力资源团队。

---
Context:
{context}

Question: {question}
"""

            # Template for condensing chat history and a new question into a standalone question
            condense_question_template_str = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

            # Create prompt templates
            QA_PROMPT = PromptTemplate.from_template(qa_template_str)
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template_str)

            # Define the chain for question generation
            question_generator_chain = LLMChain(
                llm=llm,
                prompt=CONDENSE_QUESTION_PROMPT
            )

            # Define the chain for answering the question
            qa_chain = LLMChain(
                llm=llm,
                prompt=QA_PROMPT
            )

            # Define the chain to stuff documents into the prompt
            doc_chain = StuffDocumentsChain(
                llm_chain=qa_chain,
                document_variable_name="context"
            )

            # Create the final conversational chain
            final_qa_chain = ConversationalRetrievalChain(
                retriever=retriever,
                question_generator=question_generator_chain,
                combine_docs_chain=doc_chain,
                memory=memory,
                return_source_documents=True
            )
            # --- MODIFICATION END ---

            self.finished.emit(final_qa_chain, "Status: Ready! Ask me a question about your documents.")

        except Exception as e:
            self.finished.emit(None, f"An unexpected error occurred: {e}")


# --- Part 3: PyQt5 Frontend ---

class RagAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Corporate SOP RAG Assistant")
        self.setGeometry(100, 100, 800, 600)

        # UI Elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setHtml("<h3>Welcome to your document assistant!</h3>"
                                  "<p>Initializing the system... This may take a few moments.</p>")

        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("padding: 5px;")

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your question here...")
        self.input_box.returnPressed.connect(self.handle_ask_question)  # Ask on Enter key
        self.input_box.setEnabled(False)  # Disable until RAG is ready

        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.handle_ask_question)
        self.ask_button.setEnabled(False)  # Disable until RAG is ready

        # Layout
        self.layout.addWidget(self.chat_display)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.input_box)
        self.layout.addWidget(self.ask_button)

        self.qa_chain = None
        self.setup_rag_pipeline()

    def setup_rag_pipeline(self):
        """Initializes and starts the background thread for RAG setup."""
        self.rag_thread = RagSetupThread()
        self.rag_thread.finished.connect(self.on_rag_setup_complete)
        self.rag_thread.start()

    def on_rag_setup_complete(self, qa_chain, status_message):
        """
        Callback function for when the RAG setup thread is finished.

        Args:
            qa_chain: The fully initialized ConversationalRetrievalChain object, or None if an error occurred.
            status_message: A message indicating the final status.
        """
        self.status_label.setText(status_message)

        if qa_chain:
            self.qa_chain = qa_chain
            self.input_box.setEnabled(True)
            self.ask_button.setEnabled(True)
            self.chat_display.append(f"<p style='color:green;'><b>System:</b> {status_message}</p>")
        else:
            # An error occurred, display it in the chat
            self.chat_display.append(f"<p style='color:red;'><b>System:</b> {status_message}</p>")

    def handle_ask_question(self):
        """Handles the user's question submission."""
        question = self.input_box.text().strip()
        if not question or not self.qa_chain:
            return

        self.input_box.clear()
        self.input_box.setEnabled(False)
        self.ask_button.setEnabled(False)

        # Display user's question
        self.chat_display.append(f"<p><b>You:</b> {question}</p>")
        self.status_label.setText("Status: Thinking...")
        QApplication.processEvents()  # Update the UI to show the question immediately

        # Run the QA chain in a separate thread to avoid freezing the GUI
        self.qa_thread = QThread()
        self.worker = QAWorker(self.qa_chain, question)
        self.worker.moveToThread(self.qa_thread)
        self.qa_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_answer_received)
        self.worker.finished.connect(self.qa_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.qa_thread.finished.connect(self.qa_thread.deleteLater)
        self.qa_thread.start()

    def on_answer_received(self, result):
        """Callback for when the QA chain has an answer."""
        answer = result.get("answer", "Sorry, I couldn't find an answer.")
        self.chat_display.append(f"<p><b>Assistant:</b> {answer.replace(os.linesep, '<br>')}</p>")

        # Re-enable input fields
        self.status_label.setText("Status: Ready!")
        self.input_box.setEnabled(True)
        self.ask_button.setEnabled(True)
        self.input_box.setFocus()


class QAWorker(QThread):
    """
    Worker thread to run the QA chain for a single question.
    """
    finished = pyqtSignal(dict)

    def __init__(self, qa_chain, question):
        super().__init__()
        self.qa_chain = qa_chain
        self.question = question

    def run(self):
        """Executes the QA chain and emits the result."""
        try:
            result = self.qa_chain({"question": self.question})
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({"answer": f"An error occurred while getting the answer: {e}"})


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = RagAppWindow()
    main_window.show()
    sys.exit(app.exec_())
