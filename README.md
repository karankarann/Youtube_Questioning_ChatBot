# Youtube_ChatBot

![image](https://github.com/user-attachments/assets/400b55d6-b5a7-4df7-bac2-8fae10eb1120)



## Overview

The YouTube Questioning ChatBot is designed to answer questions about any YouTube video by analyzing its transcript. This tool leverages advanced natural language processing techniques to provide accurate and relevant responses.

## Features

- **Transcript Extraction:** Fetches video transcripts using YoutubeLoader.
- **Document Processing:** Splits transcripts into chunks using RecursiveCharacterTextSplitter for efficient processing.
- **Vector Store Creation:** Uses FAISS and OpenAIEmbeddings to index transcript chunks.
- **Query Handling:** Performs similarity searches to find relevant transcript parts.
- **Response Generation:** Uses ChatOpenAI to generate detailed responses based on the relevant transcript chunks.

## Installation

1. Clone the repository:
   git clone https://github.com/karankarann/Youtube_Questioning_ChatBot.git

2. Navigate to the project directory:
   cd Youtube_Questioning_ChatBot

3. Install the required dependencies:
   pip install -r requirements.txt

4. Set up the keys in a .env file
   First, create a .env file in the root directory of the project. Inside the file, add your OpenAI API key:
   OPENAI_API_KEY="your_api_key_here"

## Usage

1. Ensure you have your OpenAI API key ready.

2. Run the script with a YouTube URL:
   python Youtube_chatbot.py --url <YouTube_URL> --api_key <Your_OpenAI_API_Key>

3. Ask questions about the video content and get detailed responses.

## Example

The bot can summarize a YouTube video or answer specific questions about it. For instance, given a YouTube URL, it can:

- Provide a summary of the video.
- Answer specific questions about the video's content.

## How It Works

- **Transcript Extraction:** The YoutubeLoader fetches the video transcript.
- **Document Processing:** The transcript is divided into chunks using RecursiveCharacterTextSplitter.
- **Vector Store Creation:** FAISS and OpenAIEmbeddings are used to create a searchable index of transcript chunks.
- **Query Handling:** The bot uses a similarity search to find the most relevant transcript chunks related to the query.
- **Response Generation:** Constructs detailed responses using the ChatOpenAI model based on the relevant transcript chunks.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
