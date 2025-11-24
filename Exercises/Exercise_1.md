## Comprehensive LangChain Coding Exercises

The exercises below draw upon the following core concepts from the sources: Model components (Language and Embedding Models), Prompt components (`PromptTemplate`, `ChatPromptTemplate`, Messages), Output Parsers (String, JSON, Pydantic), and Runnable Primitives (`RunnableSequence`, `RunnableParallel`, `RunnableLambda`, `RunnableBranch`, and LCEL chaining).

### Level 1: Easy – The Blog Post Content Generator

**Real-World Scenario:** You are building a fast, reliable internal content pipeline where users input a complex topic, and the system automatically generates a title and a concise summary suitable for a blog post introduction.

**Topics Mixed:** Language Models (Chat Models), Dynamic Prompts (`PromptTemplate`), Sequential Chaining (`RunnableSequence` via LCEL), and Basic Output Parsing (`StringOutputParser`).

**The Goal:** Construct an LCEL chain that takes a single input, `topic`, and sequentially executes three steps: Prompt Generation, LLM Call, and Output Parsing.

**Steps:**

1.  **Define a Chat Model:** Instantiate a preferred Closed Source Chat Model (e.g., GPT or Gemini).
2.  **Define the Output Parser:** Use the `StringOutputParser` to ensure the final output is clean text.
3.  **Create a Prompt Template:** Design a `PromptTemplate` that accepts the `topic` variable. The prompt should ask the model to act as a professional content writer and generate both a catchy title and a four-sentence introductory summary for a blog post about the given topic.
4.  **Build the Chain using LCEL:** Chain the components together using the pipe operator (`|`):
    $$\text{Chain} = \text{PromptTemplate} | \text{ChatModel} | \text{StringOutputParser}$$
5.  **Invoke and Review:** Invoke the chain with a specific topic (e.g., "The future of quantum computing") and observe the single, consolidated string output.

### Level 2: Medium – Customer Review Analyzer

**Real-World Scenario:** You are designing the backend service for an e-commerce platform that processes vast amounts of customer feedback. You need a fast system that simultaneously extracts structured data and assigns a creativity score to the text input.

**Topics Mixed:** Structured Output (Pydantic for schema enforcement and validation), Parallel Execution (`RunnableParallel`), Model Parameters (Temperature for creativity), and Custom Python Logic (`RunnableLambda`).

**The Goal:** Create a chain that uses `RunnableParallel` to execute two branches simultaneously: a Pydantic extraction branch (deterministic) and a sentiment/creativity analysis branch (randomized).

**Steps:**

1.  **Define a Pydantic Schema:** Create a Pydantic model (inheriting from `BaseModel`) for the structured review data. This model should include:
    *   `product_id` (String).
    *   `rating` (Integer, constrained to be between 1 and 5 using a `Field` function for validation).
    *   `pros` (List of Strings, Optional).
2.  **Create the Pydantic Chain (Deterministic Branch):**
    *   Instantiate a Chat Model, explicitly setting the **temperature parameter to 0.0** to ensure deterministic, factual output for data extraction.
    *   Attach the Pydantic parser using `with_structured_output` to enforce the schema.
    *   Define a `PromptTemplate` instructing the model to extract data from the review text according to the required schema.
    *   Chain these components to form **`Pydantic_Extractor`**.
3.  **Create the Creativity Chain (Parallel Branch):**
    *   Define a custom Python function (`def assign_creativity(review_text):`) that simulates assigning a score (e.g., 1 to 10) to the text.
    *   Convert this function into a runnable using `RunnableLambda`.
    *   Instantiate a second Chat Model, setting the **temperature parameter high (e.g., 1.2 or 1.5)** to encourage creative language usage, and ask it to provide a detailed, creative interpretation of the review.
    *   Chain this model with a `StringOutputParser` and call this branch **`Creative_Analysis`**.
4.  **Build the Parallel Chain:** Use `RunnableParallel` to execute both `Pydantic_Extractor` and `Creative_Analysis` simultaneously, receiving the original review text as input.
5.  **Invoke and Review:** Test the parallel chain with a lengthy, complex customer review. The final output should be a dictionary containing both the strictly validated Pydantic data and the creative interpretation.

### Level 3: Hard – Context-Aware Conditional Support Bot

**Real-World Scenario:** You are creating an internal documentation chatbot for your company. The bot needs to handle multi-turn conversations and conditionally route user questions. If the user asks about a general, known topic (like 'What is LangChain?'), the bot uses a simple, fast LLM call. If the user asks a specialized question requiring company context, the bot performs a semantic search on internal documents (simulated here) before answering.

**Topics Mixed:** Multi-turn Conversation (`ChatPromptTemplate`, Message Types), Conditional Logic (`RunnableBranch`), Semantic Search (Embedding Models/Document Similarity—simulated), and Flexible LCEL Flow.

**The Goal:** Construct a `RunnableBranch` that routes the conversation based on the user's latest message, using chat history for context.

**Steps:**

1.  **Define Message Types and Template:**
    *   Define System, Human, and AI Message types.
    *   Create a `ChatPromptTemplate` that includes a System Message (defining the bot's persona), a `MessagePlaceholder` for the chat history, and a Human Message (for the current query).
2.  **Define the "Router" Function (The Condition):**
    *   Create a `RunnableLambda` function (`def check_for_specialized_query(history):`) that examines the latest Human Message in the chat history.
    *   This function returns a string identifier (e.g., "GENERAL" or "SPECIALIZED"). *Simulation:* If the latest message contains keywords like "RAG" or "vector database," return "SPECIALIZED"; otherwise, return "GENERAL."
3.  **Define the "GENERAL" Chain:**
    *   Create a simple sequential chain (LCEL) that feeds the full `ChatPromptTemplate` to the Chat Model for standard Q&A.
4.  **Define the "SPECIALIZED" Chain (Simulated RAG):**
    *   *Simulate Document Retrieval:* Create a `RunnableLambda` function that mimics semantic search. If the router function indicated a specialized query, this lambda returns a pre-defined "retrieved document chunk" (e.g., "RAG applications require high-quality embeddings").
    *   Design a secondary `ChatPromptTemplate` that accepts both the `chat_history` and the `retrieved_document_chunk`. This prompt instructs the LLM to use the document chunk to answer the user's current query.
    *   Chain the simulated retrieval output with the secondary prompt and the Chat Model.
5.  **Build the Conditional Branch:** Use `RunnableBranch`:
    *   The primary input to the branch is the full chat history.
    *   The condition uses the `check_for_specialized_query` lambda.
    *   If the router returns "SPECIALIZED," run the **`SPECIALIZED` Chain**.
    *   The `RunnableBranch` default (Else) case runs the **`GENERAL` Chain**.
6.  **Full Flow:** Chain the router/condition step with the final branch output. When invoking the chain, pass in a list of messages (the `chat_history`).

This final exercise requires deep knowledge of how to create dynamic conversation structures, implement custom logic, and use conditional routing—all crucial elements for building complex Generative AI agents.