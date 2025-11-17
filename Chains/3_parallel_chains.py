from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

model1 = ChatOpenAI(model="gpt-5-nano")

model2 = ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

notes_prompt = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

quiz_prompt = PromptTemplate(
    template="Generate 5-10 quiz questions from the following text \n {text}",
    input_variables=['text'],
)

merge_prompt = PromptTemplate(
    template="Merge the provide notes and quiz questions into a single document:\nnotes -> {notes}\nquiz -> {quiz}",
    input_variables=['notes', 'quiz'],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': notes_prompt | model1 | parser,
    'quiz': quiz_prompt | model2 | parser,
})

merge_chain = merge_prompt | model1 | parser

chain = parallel_chain | merge_chain

res = chain.invoke(input={'text': text})
print(res)
chain.get_graph().print_ascii()
