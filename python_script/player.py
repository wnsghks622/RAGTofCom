from dataclasses import dataclass
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate

@dataclass
class Player:
    id: int
    llm: BaseChatModel
    prompt_template: PromptTemplate
    name: str = None
    profile: str = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"Player_{self.id}" 