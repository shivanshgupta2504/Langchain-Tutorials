## Additional Comprehensive LangChain Coding Exercises

### Level 1: Easy – The Tone-Adjusted Translator

**Real-World Scenario:** You are developing a communication tool that translates technical documentation snippets into different linguistic registers (e.g., formal business, casual social media) while preserving the core meaning.

**Topics Mixed:** Dynamic Prompting (`ChatPromptTemplate`, Message Roles), Sequential Chaining (LCEL `|`), String Output, and Model Configuration (Simulating creativity/formality via prompt instructions).

**The Goal:** Construct a flexible LCEL chain that accepts a block of text and a desired output tone (e.g., "formal," "sarcastic," "friendly"). The chain must use message roles to define the model's persona based on the tone input before generating the final output.

**Steps:**

1.  **Define the Model and Parser:** Instantiate a Chat Model (e.g., GPT/Claude) and the `StringOutputParser`.
2.  **Create a Dynamic Prompt Template:** Use `ChatPromptTemplate.from_messages` to define the input structure. This must include:
    *   A **System Message** that dynamically injects the required persona/tone using a placeholder variable (e.g., `{tone}`).
    *   A **Human Message** that includes the text to be translated/rewritten using another placeholder (e.g., `{input_text}`).
3.  **Build the Chain using LCEL:** Chain the template, model, and parser sequentially:
    $$\text{Chain} = \text{ChatPromptTemplate} | \text{ChatModel} | \text{StringOutputParser}$$.
4.  **Invoke and Review:** Invoke the chain twice with the same technical text input but use different `tone` variables (e.g., "highly professional business analyst" vs. "casual teenager") and observe the distinct outputs resulting from the dynamic System Message.

### Level 2: Medium – Legal Document Processing Pipeline

**Real-World Scenario:** A legal firm is automating the intake of new case notes. The system must clean the raw text (which often contains formatting issues) and simultaneously extract key entities and count the total number of processed words for billing purposes.

**Topics Mixed:** Custom Python Logic (`RunnableLambda`), Parallel Execution (`RunnableParallel`), Structured Output (Pydantic for validation), and Data Transformation (`RunnableSequence` components).

**The Goal:** Build a parallel chain where the raw input text is first cleaned using custom Python logic, and the cleaned output is then fed into two parallel branches: one for structured data extraction and one for custom statistical analysis.

**Steps:**

1.  **Define Pydantic Schema:** Create a Pydantic model (e.g., `CaseEntities`) to enforce the extraction of specific entities, such as `client_name` (String), `submission_date` (Date/String), and `document_type` (String, possibly constrained using a `Literal` definition).
2.  **Create the Pre-processing Lambda:** Define a standard Python function (e.g., `clean_text(raw_input)`) that removes extraneous characters (like HTML tags or excessive whitespace). Convert this function into an LCEL component using `RunnableLambda`.
3.  **Define the Extraction Chain:** Create a sequential chain (`LLM_Extractor`) using LCEL that takes the *cleaned text*, structures the output according to the Pydantic schema using the Pydantic Output Parser, and instructs the model to populate the fields.
4.  **Define the Word Counter Lambda:** Define a separate Python function (e.g., `count_words(cleaned_text)`) that simply counts the words in the input string. Convert this into a runnable using `RunnableLambda`.
5.  **Build the Final Chain:**
    *   First, pass the raw input through the **`Pre_processing_Lambda`**.
    *   Then, pipe the cleaned output into a **`RunnableParallel`** instance that executes the `LLM_Extractor` and the `Word_Counter_Lambda` simultaneously.
    $$\text{Final\_Chain} = \text{Pre\_processing\_Lambda} | \text{RunnableParallel}(\text{Extractor}, \text{Counter})$$
6.  **Invoke and Review:** Test the chain with raw, messy text. The output should be a dictionary containing both the structured, validated legal entities and the calculated word count.

### Level 3: Hard – Context-Refining RAG Bot

**Real-World Scenario:** You are building an enterprise RAG application (like a knowledge base manager or research assistant) that needs to handle complex follow-up questions effectively using a dynamic query refinement technique.

**Topics Mixed:** Conditional Logic (`RunnableBranch`), Conversational History (via `MessagesPlaceholder`), Query Refinement Chain, Simulated RAG (Retriever/Vector Store), and Complex Data Flow (`RunnablePassthrough.assign`).

**The Goal:** Create a conversational retrieval chain that inspects the length of the input message list. If it is the first message (single message), it performs RAG directly. If it is a follow-up (multiple messages), it first runs a secondary LLM chain to refine the ambiguous query into a standalone search query, and then performs RAG.

**Steps:**

1.  **Define Conversational Components:** Define the RAG components (Retriever/LLM/Parser/Prompt) required for the final answer generation (simulating the full RAG process). Ensure the final answering prompt uses a `MessagesPlaceholder` to receive the conversation history.
2.  **Define the Query Refinement Chain:** Create a specialized sequential chain (`Query_Refiner`) that takes the message history as input and instructs the LLM to generate a standalone, refined search query based on the context (e.g., transforming "Tell me more about that" into "properties of d-block elements").
3.  **Define the Router Function (The Condition):** Create a lambda function or helper that inspects the input (a list of messages, i.e., chat history). It must return `True` if the message list contains only one message (initial query) and `False` otherwise (follow-up query).
4.  **Build the Conditional Retriever:** Use **`RunnableBranch`** with the router function:
    *   **Condition True (Initial Query):** Pipe the raw current message content directly to the Retriever (simulated semantic search).
    *   **Condition False (Follow-up):** Run the `Query_Refiner` chain first, then pipe its string output (the refined query) to the Retriever.
5.  **Build the Final Chain:** Use `RunnablePassthrough.assign` to structure the final chain flow:
    *   The conditional retriever determines the relevant context.
    *   The output of the conditional retriever (the context documents) and the original `messages` list are fed to the final RAG Answer Generator.
    This architecture mimics how large systems handle conversational context without explicitly maintaining traditional LangChain Memory components, instead relying on query transformation based on message history.

This exercise requires orchestrating conditional routing based on conversational state, integrating the output of one chain (query refinement) as the input for a subsequent retrieval step, and wrapping the whole process in the dynamic flow control offered by LCEL.

***

**Analogy for LCEL Primitives:**

If traditional software development is like assembling a prefabricated house by following a blueprint step-by-step, **LangChain Expression Language (LCEL) is like working with industrial-grade Lego bricks.** Each component (Model, Prompt, Parser, Lambda) is a standardized brick that follows universal connection rules (the Runnable interface and methods like `.invoke()`). Primitives like `RunnableSequence`, `RunnableParallel`, and `RunnableBranch` are specialized connection plates and joints that allow you to dictate the flow—whether the bricks connect linearly, branch out simultaneously, or split based on a decision, enabling you to build complex, flexible structures (chains) quickly and reliably.