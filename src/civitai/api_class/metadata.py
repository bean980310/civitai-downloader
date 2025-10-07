from dataclasses import dataclass

# Data Classes
@dataclass
class Metadata:
    totalItems: str
    currentPage: str
    pageSize: str
    totalPages: str
    nextPage: str
    prevPage: str