## Detailed Summary Notes: Text Splitters in LangChain

### 1. Introduction and Definition

*   Text Splitters are the **second important component** in building RAG (Retrieval-Augmented Generation) based applications, following Document Loaders.
*   **Text Splitting** is the process of breaking large chunks of text (like articles, books, PDFs, or HTML pages) into **smaller, manageable pieces** (called chunks) that a Large Language Model (LLM) can handle effectively.
*   The code that performs this operation is called a **Text Splitter**.
*   Common strategies include dividing a PDF based on pages or paragraphs.

### 2. Why Text Splitting is Essential (Key Reasons)

Text splitting enhances the quality and efficiency of LLM-powered applications.

| Reason | Explanation |
| :--- | :--- |
| **Overcoming Model Limitations (Context Length)** | LLMs and embedding models have **maximum input size constraints** (context length limit, e.g., 50,000 tokens). If a document exceeds this limit (e.g., 100,000 words), it cannot be processed whole. Splitting allows processing these large documents. |
| **Improving Quality of Downstream Tasks** | Splitting yields better results for tasks such as embedding, semantic search, and summarization. |
| **Optimizing Computational Resources** | Working with smaller chunks is **more memory efficient** and allows for better **parallelization** of processing tasks, reducing computational requirements. |

**Quality Improvement Details:**

*   **Embeddings:** When converting text to vectors (embeddings), small chunks **better capture the semantic meaning** compared to trying to embed a massive block of text, which often results in poor quality vectors.
*   **Semantic Search:** Semantic search quality is more **precise and improved** when conducted on smaller, contextually appropriate chunks.
*   **Summarisation:** LLMs often struggle with large texts, sometimes drifting or hallucinating. Text splitting has been empirically proven to provide **better summarisation results**.

### 3. Key Text Splitting Techniques

The video discusses four main text splitting methods, ranging from simplest to most advanced:

#### A. Length-Based Text Splitting (Character Text Splitter)

*   **Principle:** This is the **simplest and fastest** technique. Chunks are created purely based on a predefined length (e.g., 100 characters or tokens).
*   **Mechanism:** The splitter traverses the text sequentially, stopping and creating a chunk once the character (or token) count is reached, regardless of linguistic boundaries.
*   **Disadvantage:** This method ignores linguistic structure, grammar, and semantic meaning. It may **cut text mid-word, mid-sentence, or mid-paragraph**. This abrupt cutting can lead to incomplete semantic meaning being captured in embeddings.

**Chunk Overlap:**

*   **Definition:** Chunk overlap specifies how many characters are shared between two consecutive chunks.
*   **Benefit:** Overlap is crucial because it helps **retain context** that might otherwise be lost when the text is abruptly cut between chunks.
*   **Trade-off:** High overlap retains more context but creates more chunks, increasing computational load.
*   **Recommendation:** For RAG applications, a chunk overlap of **10% to 20%** of the chunk size is generally recommended (e.g., 10–20 characters overlap for a 100-character chunk).

#### B. Text Structure-Based Text Splitting (Recursive Character Text Splitter)

*   **Principle:** This is **one of the most widely used** text splitting techniques. It respects the inherent hierarchical structure of text: Paragraphs (`\n\n`) $\rightarrow$ Lines/Sentences (`\n`) $\rightarrow$ Words (` `) $\rightarrow$ Characters.
*   **Mechanism:**
    1.  The algorithm uses a predefined list of separators.
    2.  It attempts to split the text using the **highest level separator** (e.g., paragraph `\n\n`).
    3.  If the resulting chunks are still larger than the allowed `chunk_size`, the algorithm **recursively** attempts to split those large chunks using the next separator (e.g., line/sentence `\n`).
    4.  It continues this process down to the word and character level until the size constraint is met.
    5.  It also attempts to **merge smaller chunks** if the resulting merged chunk remains below the `chunk_size` limit (optimization).
*   **Advantage:** This method actively tries to avoid splitting mid-sentence or mid-word, making the resulting chunks much more **contextually coherent** than those created by the Character Text Splitter.

#### C. Document-Based Text Splitting (Recursive Splitter for Code/Markdown)

*   **Principle:** This is an **extension** of the Recursive Character Text Splitting idea, applied to documents that are not standard plain text, such as **code** (Python, Java, etc.) or **markup languages** (Markdown, HTML).
*   **Mechanism:** Since code and markdown are not structured using paragraphs and sentences, this technique uses the same recursive algorithm but substitutes the default separators with **language-specific separators** (constructs).
    *   *Example (Python):* Separators include keywords like `class` or `def` (for functions), followed by normal separators (paragraph, line, word, character).
*   **Advantage:** It ensures that contextually related code blocks (like a class definition or a function) remain within a single chunk, which is essential for LLMs processing code.

#### D. Semantic Meaning-Based Text Splitting (Experimental)

*   **Principle:** Splitting decisions are made based on the **semantic meaning** of the text, not length or structure. This addresses issues where a single paragraph discusses two entirely different topics (e.g., agriculture and IPL), which would normally result in poor embedding quality if left unsplit.
*   **Mechanism (Sliding Window Approach):**
    1.  The text is broken down into small units, typically sentences.
    2.  An embedding model (e.g., OpenAI's) generates vector embeddings for each sentence.
    3.  The **similarity** (e.g., Cosine Similarity) between the vectors of consecutive sentences (S1 and S2, S2 and S3, etc.) is calculated.
    4.  A **sudden, abrupt drop in similarity** between a pair of consecutive sentences indicates a topic change, signaling where the split should occur.
    5.  A **threshold** is used to define how significant the drop must be (e.g., using **Standard Deviation**, Percentile, or Interquartiles).
*   **Status:** While this concept is very promising for the future, the LangChain implementation (`SemanticChunker`) is currently **experimental** and may not yet provide consistently accurate results.
*   **Overall Best Choice:** As of now, the **Recursive Character Text Splitter** is the recommended and most commonly used method.

---

## Code Examples Provided in the Video

The following code snippets demonstrate the implementation of the discussed text splitters using LangChain components:

### 1. Character Text Splitter (Basic Splitting)

This example shows how to import and use the basic splitter, setting the chunk size and overlap.

```python
from langchain.text_splitter import CharacterTextSplitter

# 1. Define the text to split (Example text omitted for brevity)
text = "..." 

# 2. Create the Splitter Object
splitter = CharacterTextSplitter(
    chunk_size=100,  # Size of each chunk in characters
    chunk_overlap=0, # No overlap for this basic example
    separator=""     # Split after reaching the chunk size limit
)

# 3. Perform the splitting
chunks = splitter.split_text(text)

# Example output (e.g., print(chunks))
```

### 2. Character Text Splitter with Document Loaders

This shows how to connect the Document Loader workflow with the Text Splitter, specifically using `split_documents` to handle loaded document objects.

```python
# Assuming installation of required packages for PDF loading
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# 1. Load Documents from a PDF file
loader = PyPDFLoader("DL_curriculum.pdf")
docs = loader.load() # Docs is a list of Document objects (one per page)

# 2. Define Splitter (Chunk size 200 used in demonstration)
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)

# 3. Split the Document objects
# Uses split_documents() function instead of split_text()
split_docs = splitter.split_documents(docs)

# Example output (e.g., print(split_docs.page_content))
```

### 3. Recursive Character Text Splitter

This is the most highly recommended splitter, often used with a decent chunk size.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Define the text to split (Example text omitted for brevity)
text = "..." 

# 2. Create the Splitter Object
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # Starts trying to break based on structural hierarchy
    chunk_overlap=0 
)

# 3. Perform the splitting
chunks = splitter.split_text(text)

# Example output (e.g., print(len(chunks)))
```

### 4. Recursive Character Text Splitter for Specific Languages (Document-Based)

This demonstrates splitting Python code, utilizing language-specific constructs as separators.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# 1. Define the Python code text (Example code omitted for brevity)
python_code_text = "..." 

# 2. Create the Splitter Object using the 'from_language' method
# This tells the splitter to use Python-specific separators (class, def, etc.)
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=350,          # Example chunk size
    chunk_overlap=50         # Example chunk overlap
)

# 3. Perform the splitting
chunks = splitter.split_text(python_code_text)

# Example output (e.g., print(chunks))
# The same approach works for Language.MARKDOWN, Language.JAVASCRIPT, etc.
```

### 5. Semantic Meaning-Based Text Splitting (Experimental)

This example uses `SemanticChunker`, which relies on an embedding model (like OpenAI) and a statistical threshold to determine topic changes.

```python
# Note: SemanticChunker is currently in the experimental library stage
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings # Requires an embedding model

# 1. Define the sample text (Text with mixed topics omitted for brevity)
sample_text = "..."

# 2. Initialize the Embedding Model
embeddings = OpenAIEmbeddings()

# 3. Create the Splitter Object
splitter = SemanticChunker(
    embeddings=embeddings,
    # Specifies the method used to detect context change boundary
    breakpoint_threshold_type="standard_deviation" 
    # breakpoint_threshold_amount (e.g., 1, 2, 3 standard deviations) can also be set
)

# 4. Perform the splitting
chunks = splitter.split_text(sample_text)

# Example output (e.g., print(len(chunks)))
```