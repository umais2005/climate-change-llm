DIRECTORY_PATH = "./data/"
INDEX_NAME = "test"

import io
import json
import os
import streamlit as st
import feedparser
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from warnings import filterwarnings
from moviepy.editor import AudioFileClip
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pydub import AudioSegment
import unicodedata
from openai import OpenAI
filterwarnings("ignore")
import time
import re
from dotenv import load_dotenv


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TEMP_DOWNLOAD_DIR = './temp_downloads'


class DocumentProcessor:
    def __init__(self, directory_path="./data/",
                index_name="test",
                drive_folder_id ="1-50Qfs8g8DELeudumJNarNrS4WRA6-F6",
                rss_url = "https://feeds.simplecast.com/XFfCG1w8"):
        load_dotenv()
        self.rss_url = rss_url
        self.drive_folder_id = drive_folder_id
        self.dimensions = 1536
        self.directory = directory_path
        self.index_name = index_name
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
        self.vector_store = self.load_pinecone_vector_store()
        print("Document Processor initialized.")

    def load_pinecone_vector_store(self):
        # pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_api_key = "bc6855e9-edf3-44ca-9608-ea06f2e9d8b8"
        if not pinecone_api_key:
            raise ValueError("No Pinecone API key found in environment variables.")
        
        pc = Pinecone(api_key=pinecone_api_key)

        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
                
        self.index = pc.Index(self.index_name)
        print(f"Pinecone vector store '{self.index_name}' loaded.")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        return self.vector_store
    

    def get_file_paths_from_directory_and_subdirectories(self):
        """
        Get all PDF and DOCX file paths from the directory,
        including files in nested directories.
        """
        file_paths = []
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith((".pdf", ".docx")):
                    # Construct the full file path and add it to the list
                    file_paths.append(os.path.join(root, filename))
        return file_paths


    def check_existing_docs_by_id(self, doc_ids):
        """
        Check if the document IDs (filenames) exist in the Pinecone index.
        """
        # print(doc_ids)
        existing_ids = set()
        for doc_id in doc_ids:
            try:
                response = self.index.fetch(ids=[doc_id])
                # print(response)
                if response["vectors"]:
                    existing_ids.add(doc_id)
            except Exception as e:
                print(f"Error checking document ID {doc_id}: {str(e)}")
        return existing_ids
    
    
    def authenticate_drive_with_service_account(self):
        service_account_info = st.secrets["google_drive"]["service_account_info"]
        service_account_info = json.loads(service_account_info)
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)


    def download_file_from_drive(self, service, file_id, file_name):
        os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
        destination = os.path.join(TEMP_DOWNLOAD_DIR, file_name)
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(destination, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Fetching {int(status.progress() * 100)}% of {file_name}.")
        return destination
    

    def list_files_in_drive(self, service, folder_id):
        """Returns list of all files in a given folder and its subfolders each
        as dict with values:
        1.mimeType
        2. id
        3. name
        """
        files = []
        folder_queue = [self.drive_folder_id]
        while folder_queue:
            current_folder_id = folder_queue.pop(0)
            query = f"'{current_folder_id}' in parents"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            items = results.get('files', [])
            for item in items:
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    folder_queue.append(item['id'])
                else:
                    files.append(item)
        return files

    def process_and_add_documents_from_drive(self):
        service = self.authenticate_drive_with_service_account()
        files = self.list_files_in_drive(service, self.drive_folder_id)

        if not files:
            print("No documents found in the specified folder.")
            st.warning("No documents found in the specified folder.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)

        # Step 1: Extract filenames (without extensions) to use as IDs
        filenames = [os.path.splitext(file['name'])[0] for file in files]
        
        # Step 2: Check if the document IDs already exist in Pinecone
        existing_ids = self.check_existing_docs_by_id(filenames)

        # Step 3: Filter out files that already exist in Pinecone
        new_files = [file for file in files if os.path.splitext(file['name'])[0] not in existing_ids]

        if not new_files:
            print("No new documents to add.")
            st.success("No new documents to add.")
            return
        else:
            st.success("New files found. Fetching:")
        # Step 4: Process and add only the new files to Pinecone
        for file in new_files:
            file_id = file['id']
            file_name = file['name']
            st.write(f"Fetching file from Drive: {file_name}")

            # Download the file temporarily
            downloaded_path = self.download_file_from_drive(service, file_id, file_name)

            # Load the file using the appropriate loader
            if file_name.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path=downloaded_path)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file_path=downloaded_path)
            else:
                print(f"Unsupported file format: {file_name}")
                continue

            filename = os.path.splitext(file_name)[0]
            # print(f"Processing document: {filename}")
            st.write(f"Processing document: {filename}")
            
            # Add document-level dummy vector
            dummy_vector = [1.0] * 1536  # Ensure vector values are floats
            self.index.upsert(vectors=[(filename, dummy_vector)])  # Add document-level vector for tracking

            # Load and split the document into chunks
            file_docs = loader.load()
            chunks = text_splitter.split_documents(file_docs)
            print(f"Processed {len(chunks)} chunks from document {filename}")

            # Generate IDs for the chunks
            ids = [f"{filename}_chunk_{i}" for i, _ in enumerate(chunks)]
            
            # Add chunk-level vectors using add_documents
            self.vector_store.add_documents(documents=chunks, ids=ids)
            st.write(f"Document processing and vector store update complete for {filename}.")
            print(f"Document processing and vector store update complete for {filename}.")

            # Clean up the downloaded file
            os.remove(downloaded_path)


    def process_and_add_documents_from_local(self):
        """
        Process new PDF and DOCX documents from the directory and add them to Pinecone.
        Only new documents (not already in Pinecone) are processed.
        """
        # Step 1: Get all PDF and DOCX file paths from the directory and its subdirectories
        file_paths = self.get_file_paths_from_directory_and_subdirectories()

        # Step 2: Extract filenames (without extensions) to use as IDs
        filenames = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]

        # Step 3: Check if the document IDs already exist in Pinecone
        existing_ids = self.check_existing_docs_by_id(filenames)

        # Step 4: Filter out paths of files that already exist
        new_file_paths = [path for path in file_paths if os.path.splitext(os.path.basename(path))[0] not in existing_ids]

        if not new_file_paths:
            print("No new documents to add.")
            return

        # Initialize text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)

        # Step 5: Process new files
        for file_path in new_file_paths:
            # Load PDF or DOCX file
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path=file_path)
                print(f"Processing PDF: {file_path}")
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path=file_path)
                print(f"Processing DOCX: {file_path}")
            else:
                print(f"Unsupported file format: {file_path}")
                continue
            filename = os.path.splitext(os.path.basename(file_path))[0]
            print(filename)
            # Step 5: Add document-level dummy vector
            dummy_vector = [1.0] * 1536  # Ensure vector values are floats
            self.index.upsert(vectors=[(filename, dummy_vector)])  # Add document-level vector for tracking

            # Step 6: Load and split the document into chunks
            file_docs = loader.load()
            chunks = text_splitter.split_documents(file_docs)
            print(f"Processed {len(chunks)} chunks from document {filename}")

            # Generate IDs for the chunks
            ids = [f"{filename}_chunk_{i}" for i, _ in enumerate(chunks)]
            
            # Step 7: Add chunk-level vectors using add_documents
            self.vector_store.add_documents(documents=chunks, ids=ids)
            print("Document processing and vector store update complete.")

    def get_podcasts(self):
        """
        Fetches podcasts from an RSS feed and extracts MP3 URLs.
        
        Parameters:
        self.rss_url (str): URL of the RSS feed.
        
        Returns:
        list: A list of dictionaries with new podcast details including title, published date, and MP3 URL.
        """
        print("Getting RSS Feed")
        feed = feedparser.parse(self.rss_url)
        new_podcasts = []

        for entry in feed.entries:
            mp3_url = next(
                (enclosure.href for enclosure in entry.get('enclosures', [])
                if enclosure.type == 'audio/mpeg'), None
            )
            title = entry.title.strip()
            title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
            if mp3_url:
                podcast_info = {
                    'title': title,
                    'published': entry.published.strip() if 'published' in entry else entry.updated,
                    'mp3_url': mp3_url
                }
                new_podcasts.append(podcast_info)

        return new_podcasts

    def split_audio_with_pydub(input_path, output_dir, chunk_duration=1200):
        """
        Splits the audio file into chunks using pydub, with parallel processing.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio = AudioSegment.from_file(input_path)
        total_duration = len(audio)  # Duration in milliseconds

        def save_chunk(start_time, end_time, chunk_index):
            chunk = audio[start_time:end_time]
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}.mp3")
            chunk.export(chunk_path, format="mp3")

        # Split audio into chunks and save each in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, total_duration, chunk_duration * 1000):  # chunk_duration is in seconds, convert to ms
                start_time = i
                end_time = min(i + chunk_duration * 1000, total_duration)
                chunk_index = i // (chunk_duration * 1000)
                futures.append(executor.submit(save_chunk, start_time, end_time, chunk_index))

            # Wait for all chunks to finish processing
            for future in futures:
                future.result()


    def split_audio_with_moviepy(self, input_path, output_dir, chunk_duration=1200):
        """
        Splits the audio file into chunks using moviepy.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_clip = AudioFileClip(input_path)
        total_duration = audio_clip.duration

        # Split audio into chunks and save each as a separate file
        for i in range(0, int(total_duration), chunk_duration):
            start_time = i
            end_time = min(i + chunk_duration, total_duration)
            chunk = audio_clip.subclip(start_time, end_time)
            chunk_path = os.path.join(output_dir, f"chunk_{i//chunk_duration:03d}.mp3")
            chunk.write_audiofile(chunk_path, codec='mp3')

    def transcribe_chunk(self, chunk_path):
        """
        Transcribes a single audio chunk using OpenAI Whisper API.
        """
        with open(chunk_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        os.remove(chunk_path)  # Clean up chunk file after transcription
        return transcription.text  # Collect only text

    def process_podcast_audio(self, audio_url, chunk_duration=1200):
        """
        Downloads, segments, and transcribes an audio file using OpenAI Whisper API.
        """
        output_dir = os.path.join(TEMP_DOWNLOAD_DIR, "chunks")
        self.split_audio_with_moviepy(audio_url, output_dir, chunk_duration=chunk_duration)

        transcripts = []

        # Use ProcessPoolExecutor to transcribe each chunk in parallel
        with ThreadPoolExecutor() as executor:
            # Submit all transcriptions tasks to the pool
            futures = {
                executor.submit(self.transcribe_chunk, os.path.join(output_dir, chunk_filename)): chunk_filename
                for chunk_filename in sorted(os.listdir(output_dir))
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                transcripts.append(future.result())

        os.rmdir(output_dir)  # Remove the chunks directory after all transcriptions are done
        return " ".join(transcripts)


    def add_podcast_to_index(self, podcast_id, transcript):
        """
        Splits the transcript into smaller chunks, converts each to a Document,
        and adds them to the Pinecone index.
        """
        self.index.upsert([(podcast_id, [1.0] * self.dimensions)])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript)
        documents = [Document(page_content=chunk, metadata={"source": podcast_id}) for chunk in chunks]
        self.vector_store.add_documents(documents=documents, ids=[f"{podcast_id}_chunk_{i}" for i in range(len(documents))])

    def process_and_add_new_podcasts(self, latest_n=-1):
        """
        Main method to retrieve, process, and add new podcasts from RSS feed.
        """
        print("Fetching podcasts from RSS feed...")
        st.success("Fetching podcasts from RSS feed...")
        podcasts = self.get_podcasts()

        podcast_ids = [podcast["title"] for podcast in podcasts]
        st.success("Checking dupes for podcasts")
        existing_ids = self.check_existing_docs_by_id(podcast_ids)
        new_podcasts = [podcast for podcast in podcasts if podcast['title'] not in existing_ids][:latest_n]
        print(new_podcasts)
        if new_podcasts:
            st.success("New podcasts found.")
            print("Processing podcasts texts")
            for podcast in new_podcasts:
                podcast_id = podcast["title"]
                print(podcast_id)
                transcript = self.process_podcast_audio(podcast["mp3_url"])
                self.add_podcast_to_index(podcast_id, transcript)
                st.success(f"Podcast '{podcast_id}' processed and added to Pinecone.")
        else:
            st.success("No new podcasts to be ingested")
    
# Example usage:
if __name__ == "__main__": ## TESTING ##    
    # folder_id = "1ptf4zFnw0Lu5sJMMDIqwY2WxDVj29kz1"
    processor = DocumentProcessor(INDEX_NAME)
    # processor.process_and_add_documents_from_drive()
    # processor.process_and_add_documents_from_local()
    text = processor.process_podcast_audio("https://chrt.fm/track/DBDAF8/cdn.simplecast.com/audio/a40ab607-6c05-44d6-b237-5cd4abd47ac6/episodes/5b62ad45-1287-402e-b50a-0dcf220a1760/audio/2adb5029-5fb7-4afe-bec3-4a5f48358581/default_tc.mp3?aid=rss_feed&feed=XFfCG1w8")
    # existing_ids = processor.check_existing_docs_by_id(files)

    # processor.process_and_add_documents_from_local()
    # filepaths = processor.get_file_paths_from_directory_and_subdirectories()
    # print(filepaths)
    # processor.extract_filenames_from_vector_ids(filenames)
    #     print(processor.check_existing_docs_by_id(filenames))
