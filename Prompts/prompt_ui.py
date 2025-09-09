from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.2)

st.header("Research Tool")

paper_input = st.selectbox(
    label="Select Research Paper Name",
    options=[
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    label="Select Explanation Style",
    options=[
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    label="Select Explanation Length",
    options=[
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

# Creating a Dynamic Template
# template = PromptTemplate(
#     template="""
#     Please summarize the research paper titled "{paper_input}" with the following specifications:
#     Explanation Style: {style_input}
#     Explanation Length: {length_input}
#     1. Mathematical Details:
#        - Include relevant mathematical equations if present in the paper.
#        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
#     2. Analogies:
#        - Use relatable analogies to simplify complex ideas.
#     If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
#     Ensure the summary is clear, accurate, and aligned with the provided style and length.
#     Give the response in markdown format.
#     """,
#     input_variables=["paper_input", "style_input", "length_input"],
#     validate_template=True
# )

# Loading the template from json file
template = load_prompt("template.json")

# Fill the placeholders in the template
prompt = template.invoke(input={
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

# user_input = st.text_input("Enter your Prompt") # This is static prompt

if st.button("Summarize"):
    # The Chain mechanism
    # chain = template | model
    # result = chain.invoke(input={
    #     "paper_input": paper_input,
    #     "style_input": style_input,
    #     "length_input": length_input
    # })

    result = model.invoke(prompt)
    # st.write(result.content)
    st.markdown(result.content)
