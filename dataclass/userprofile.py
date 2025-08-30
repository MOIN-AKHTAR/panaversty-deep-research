from dataclasses import dataclass, field
from typing import List

@dataclass
class UserProfile:
    name: str = "Moin"
    interests: List[str] = field(default_factory=lambda: [])
    results_count: int = 3
