import sys

sys.path.insert(0, r"./")
from .base import BaseConfig
from typing import List, Dict
from dataclasses import dataclass, asdict, fields


@dataclass
class COTConfig(BaseConfig):
    """
    For the COT dataset
    """

    question: str
    reasoning: str
    answer: str = None

    @property
    def __repr__(self) -> str:
        s = ""
        s += f"\n Question id: {self.qas_id}"
        s += f"\n Question: {self.question}"
        if self.reasoning:
            s += f"\n Reasoning: {self.reasoning}"
        if self.answer:
            s += f"\n Answer text: {self.answer}"
        return s

    @property
    def get_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def get_keys(cls) -> List[str]:
        all_fields = fields(cls)
        return [v.name for v in all_fields]


if __name__ == "__main__":
    print("TESTTEST")
