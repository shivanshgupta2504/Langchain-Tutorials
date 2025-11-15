from langchain_core.prompts import load_prompt

template = load_prompt("template.json")
# print(template.input_variables)

prompt = template.invoke(input={
    "paper_input": "Attention Is All You Need",
    "style_input": "Beginner-Friendly",
    "length_input": "Short (1-2 paragraphs)"
})
print(prompt.text)
