from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# Schema
class Review(BaseModel):
    key_themes: List[str] = Field(description="Write down all the key themes discussed in the review in a list.")
    summary: str = Field(description="A brief summary of the Review")
    sentiment: Literal["pos", "neg"] = Field(description="Return Sentiment of the review.")
    pros: Optional[List[str]] = Field(description="Write all the Pros inside a list", default=None)
    cons: Optional[List[str]] = Field(description="Write all the Cons inside a list", default=None)
    name: Optional[str] = Field(description="Return name of the Reviewer", default=None)


structured_model = model.with_structured_output(Review)  # This will provide the structured output on invoke

result = structured_model.invoke(
    """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Shivansh Gupta
    """
)

result = dict(result)
for key in result.keys():
    print(f"{key}: {result[key]}")
