# talent_faq_chatbot

An RAG (Retrieval-Augmented Generation) Agent built using LangGraph and LlamaIndex. This chatbot leverages advanced retrieval techniques and LLM capabilities to provide accurate and context-aware responses to user inquiries.

## Key Features
- **API Integration**: Utilizes the OpenAI API to generate responses.
- **Enhanced Retrieval System**: Combines **Milvus**, **BM25**, and an **LLM reranker** to deliver precise and contextually relevant retrieval results.

## To-Do List

### 1. Develop a Frontend for the RAG API
To provide an interactive user experience, a frontend interface needs to be developed to utilize the RAG API. Recommended GitHub repositories for frontend development:

- **Vue.js**: [ChatGPT UI](https://github.com/WongSaang/chatgpt-ui)
- **Next.js**: [ChatGPT Next Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)

These repositories offer ready-made templates and user-friendly interfaces that can be customized to work with the RAG API.

---

### 2. Update the BM25 Retriever to Use a Docstore
To improve retrieval efficiency and better organize document storage, update the BM25 retriever to integrate with a docstore. This enhancement will allow for faster and more structured document searches.

Reference documentation for guidance:
- **LlamaIndex BM25 Retriever**: [LlamaIndex BM25 Retriever Guide](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/)

By incorporating a docstore, retrieval performance will be significantly improved, especially in large datasets.

---

With these updates, the Talent FAQ Chatbot will achieve better scalability, faster response times, and enhanced user experience through a modernized frontend and more efficient document storage and retrieval.


## Building the Docker Images
To build the Docker images for the application, use the following command:
```bash
docker compose build
```

## Configuration
Before running the Docker containers, make sure to modify the configuration files to suit your requirements:

1. **`nodes_api_config.json`**: Configure the LLM (Large Language Model) settings.
2. **`env_config.json`**: Configure the embedding model and other environment-specific variables.

These configuration files are located in the `config` directory:
```
config/
  |- nodes_api_config.json
  |- env_config.json
```

## Running the Docker Containers

### Start All Containers
To start all containers in detached mode:
```bash
docker compose up -d
```

### Run a Specific Container
To start a specific container, replace `<container_name>` with the desired container's name:
```bash
docker compose up <container_name>
```

## Additional Notes
- Ensure that Docker and Docker Compose are properly installed on your system.
- If any changes are made to the configuration files, rebuild the images before starting the containers using `docker compose build`.

