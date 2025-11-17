## Detailed Summary Notes on LangChain Runnables

Runnables are a core concept in LangChain designed to standardize components and enable flexible, complex workflow construction.

### The Necessity of Runnables

Historically, LangChain components (like Prompt Templates, LLMs, Parsers, and Retrievers) were developed independently and were **not standardized**.

1.  **Lack of Standardization:** Different components required different methods for interaction:
    *   Prompt Templates used `format`.
    *   LLMs used `predict`.
    *   Parsers used `parse`.
    *   Retrievers used `get_relevant_documents`.
2.  **Connection Difficulty:** This lack of a standard interface made it difficult to connect components seamlessly and create flexible workflows.
3.  **The Chains Problem:** To connect components, LangChain introduced **Chains** (e.g., `LLMChain`, `RetrievalQAChain`). However, creating a custom Chain function for every potential use case led to **too many chains**, making the codebase large and the learning curve steep for new AI engineers.
4.  **The Solution:** LangChain identified the need to **standardize all components**. This was achieved by introducing the concept of **Runnables**.

### Core Characteristics of Runnables

Runnables bring standardization by adhering to a common interface, allowing components to be easily connected:

1.  **Unit of Work:** Every Runnable represents a unit of work that receives an input, processes it, and returns an output.
2.  **Common Interface:** All Runnables follow the same set of rules and methods.
3.  **The `invoke` Method:** The primary standardized method used to communicate with any Runnable is **`invoke`**. Other methods include `batch` (for processing multiple inputs simultaneously) and `stream` (for streaming output).
4.  **Seamless Connection:** When Runnables are connected, the output of the first Runnable automatically serves as the input for the next.
5.  **Chain is a Runnable:** A workflow or chain formed by connecting multiple Runnables is itself a Runnable, allowing complex structures to be built from smaller workflows.

> Runnables can be conceptually understood as **Lego Blocks**: regardless of their function or shape, they all follow a common connection interface, allowing developers to connect them in flexible ways to build any structure (workflow).

### Types of Runnables

Runnables are divided into two main categories:

| Category | Description | Example Components |
| :--- | :--- | :--- |
| **Task-Specific Runnables** | Core LangChain components that have been converted into Runnables. They have a specific purpose (e.g., designing prompts, interacting with LLMs). | `ChatOpenAI` (model), `PromptTemplate`, `Retriever`. |
| **Runnable Primitives** | Fundamental building blocks that help orchestrate the execution logic by defining how other Runnables interact (sequentially, parallelly, or conditionally). | `RunnableSequence`, `RunnableParallel`, `RunnableBranch`, `RunnableLambda`, `RunnablePassthrough`. |

### Key Runnable Primitives (Runeable Primitives)

| Primitive | Purpose | Key Feature |
| :--- | :--- | :--- |
| **`RunnableSequence`** | Connects two or more Runnables in a **sequential pipeline**. This is the most frequently used primitive. | The output of the preceding Runnable becomes the input of the next. |
| **LangChain Expression Language (LCEL)** | A declarative and cleaner way to define sequential chains. | Uses the **pipe operator (`|`)** for sequential chaining (e.g., `R1 | R2 | R3`). |
| **`RunnableParallel`** | Executes multiple Runnables or Chains **simultaneously** (in parallel). | **All parallel branches receive the same input**. The output is a **dictionary** where keys map to the branch outputs. |
| **`RunnablePassthrough`** | A special primitive that returns the input it receives **as it is**, performing no processing. | Useful in parallel chains to **preserve intermediate output** alongside the results of subsequent processing. |
| **`RunnableLambda`** | Converts **any standard Python function** into a Runnable. | Allows custom logic (like pre-processing or data manipulation) to be integrated seamlessly into a chain. |
| **`RunnableBranch`** | Used to create **conditional chains**; serves as the "if-else statement" of LangChain. | Takes conditions and corresponding Runnables in tuples. Only one branch is triggered based on the conditional logic. |

---

## Code Examples

The following code snippets illustrate the use of key Runnable Primitives using actual LangChain components (as demonstrated in the sources), often requiring necessary imports such as `ChatOpenAI`, `PromptTemplate`, `StringOutputParser`, and the relevant `Runnable` classes.

### 1. Simple Sequential Chain using LCEL (Preferred Syntax)

The easiest way to define a sequential chain is by using the LangChain Expression Language (LCEL) pipe operator (`|`):

```python
# Necessary Imports (Implicit: load_dotenv, ChatOpenAI, PromptTemplate, StringOutputParser)

# Define Components (Task-Specific Runnables)
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)
model = ChatOpenAI()
parser = StringOutputParser()

# Define the Chain using LCEL (R1 | R2 | R3)
chain = prompt | model | parser

# Invoke the Chain
result = chain.invoke({"topic": "AI"})
# Prints the joke as a string
```
*(Note: LCEL replaces the explicit use of `RunnableSequence([...])` for standard sequential chains)*.

### 2. Parallel Chain (`RunnableParallel`)

This example generates a tweet and a LinkedIn post simultaneously based on the same input topic:

```python
from langchain.schema.runnable import RunnableParallel, RunnableSequence

# Define Prompts
prompt_tweet = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)
prompt_linkedin = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=["topic"]
)

# Define Model and Parser (reused from above)
model = ChatOpenAI()
parser = StringOutputParser()

# Define Sequential Chains for each branch
tweet_chain = RunnableSequence([prompt_tweet, model, parser])
linkedin_chain = RunnableSequence([prompt_linkedin, model, parser])

# Combine into a Parallel Chain
parallel_chain = RunnableParallel({
    "tweet": tweet_chain,
    "linkedin": linkedin_chain
})

# Invoke the Parallel Chain
result = parallel_chain.invoke({"topic": "AI"}) 
# Result is a dictionary: {'tweet': '...', 'linkedin': '...'}
```

### 3. Using `RunnablePassthrough` to Preserve Intermediate Output

This is useful for getting the original text output (e.g., the joke) alongside a processed version (e.g., the explanation):

```python
from langchain.schema.runnable import RunnablePassthrough

# Define the Explanation Sequential Chain (explanation_chain = prompt_2 | model | parser)
# Assuming joke_generation_chain (R1 | R2 | R3) already created

# Define the Parallel Step
parallel_step = RunnableParallel({
    # Path 1: Preserve the input (the joke text)
    "joke": RunnablePassthrough(),
    # Path 2: Process the input to get the explanation
    "explanation": explanation_chain
})

# Final Chain (Sequential connection: Joke Generation -> Parallel Step)
final_chain = joke_generation_chain | parallel_step

# Invoke the Chain
result = final_chain.invoke({"topic": "Cricket"}) 
# Result is a dictionary: {'joke': '...', 'explanation': '...'}
```

### 4. Integrating Custom Logic with `RunnableLambda`

This converts a Python function into a Runnable to count the words in the generated joke:

```python
from langchain.schema.runnable import RunnableLambda

# Define a Python function for custom logic
def word_count(text):
    return len(text.split())

# Define the Parallel Step (assuming joke_generation_chain created)
parallel_step = RunnableParallel({
    "joke": RunnablePassthrough(),
    # Convert the function into a Runnable
    "word_count": RunnableLambda(word_count) 
    # Alternatively, use a lambda function directly: 
    # "word_count": RunnableLambda(lambda x: len(x.split()))
})

# Final Chain (Joke Generation -> Parallel Step)
final_chain = joke_generation_chain | parallel_step

# Invoke the Chain
result = final_chain.invoke({"topic": "AI"})
# Result is a dictionary: {'joke': '...', 'word_count': 5}
```

### 5. Conditional Execution using `RunnableBranch`

This checks if a report (text) is over 500 words. If true, it summarizes it (using a summary chain); if false, it passes it through (prints as is):

```python
from langchain.schema.runnable import RunnableBranch

# Define Runnables (model, parser, report_gen_chain, summary_chain)
# ... assuming setup

# Define the If condition (checks word count)
condition = lambda x: len(x.split()) > 500

# Define the Runnable Branch
branch_chain = RunnableBranch(
    # Tuple 1: (Condition, Runnable_if_True)
    (condition, summary_chain), 
    
    # Default (Else) condition: Runnable_if_False
    RunnablePassthrough() 
)

# Final Chain
final_chain = report_gen_chain | branch_chain

# Invoke the Chain
result = final_chain.invoke({"topic": "Russia vs Ukraine"})
# If condition is true (e.g., > 300 words), summary_chain output is returned.
# If condition is false, the full report text is returned by RunnablePassthrough.
```

---

## Practice Questions for Revision

### Conceptual Questions

1.  **Standardization vs. Flexibility:** Explain the core problem LangChain faced regarding non-standardized components (like `LLM.predict` versus `PromptTemplate.format`) and how the introduction of the Runnable interface solved this issue.
2.  **The Purpose of `invoke`:** What is the significance of the `invoke` method in the Runnable interface, and what legacy methods did it standardize and eventually replace?.
3.  **Categories of Runnables:** Differentiate between **Task-Specific Runnables** and **Runnable Primitives**, providing examples of each.
4.  **LCEL:** What does LCEL stand for, and what is the primary primitive it simplifies? Describe the syntax used in LCEL.
5.  **Inputs and Outputs of Parallel:** When using `RunnableParallel`, what is the guarantee regarding the input received by all executing branches, and what is the required format of the output from `RunnableParallel`?.
6.  **`RunnablePassthrough` Role:** Why is `RunnablePassthrough` necessary, given that it performs no processing? Describe a specific scenario where its use is critical.
7.  **Chain Composition:** Explain the principle that a chain of Runnables is itself a Runnable, and why this is a powerful feature for building complex applications.

### Coding and Implementation Questions

For the following coding questions, assume all necessary LangChain components (`PromptTemplate`, `ChatOpenAI`, `StringOutputParser`) and Runnable primitives have been imported and initialized.

1.  **Sequential Chain Practice (LCEL):**
    You have three components: `R_topic_prompt`, `R_model`, and `R_parser`. Write the most concise Python code (using LCEL) to link them sequentially and invoke the chain with the input `{"topic": "Physics"}`.

2.  **Parallel Execution Practice:**
    You need to take a user's query and send it to two separate LLM pipelines simultaneously. One pipeline (`R_chain_A`) should translate the query into Spanish, and the other (`R_chain_B`) should categorize the query (e.g., "sales" or "support"). Write the code to define a parallel chain that executes both processes and names the outputs 'translation' and 'category'.

3.  **Custom Data Cleaning (`RunnableLambda`):**
    A review text input often contains leading/trailing whitespace. Write a small Python function `clean_text(text)` that strips this whitespace and converts the text to lowercase. Then, define a chain that uses `RunnableLambda` to incorporate this function before sending the cleaned text to an LLM (`R_model`) for sentiment analysis.

4.  **Preserving Input (`RunnablePassthrough`):**
    A sequential chain (`R_summary_chain`) generates a summary from a long document. You want the final output to contain both the original document and the generated summary. Define the necessary parallel step using `RunnablePassthrough` to achieve this, naming the outputs 'original\_doc' and 'summary'.

5.  **Conditional Workflow (`RunnableBranch`):**
    A sequential chain (`R_sentiment_check`) determines the sentiment of a customer review and returns either 'Positive' or 'Negative' (as a string).

    *   If the sentiment is 'Positive', a separate Runnable (`R_thank_you_chain`) should be executed to draft a thank-you response.
    *   If the sentiment is 'Negative', a different Runnable (`R_escalation_chain`) should be executed to alert the support team.

    Define the `RunnableBranch` required for this conditional logic.

6.  **Multi-Chain Composition:**
    Based on the answer to question 5, define the `final_workflow` that first runs `R_sentiment_check` and then uses the result to trigger the `RunnableBranch` from question 5. Use LCEL for the final composition.