"""
Custom domain exceptions for the CiteBase application.
"""


class CiteBaseError(Exception):
    """Base exception for all CiteBase domain errors."""
    pass


class TenantNotFoundError(CiteBaseError):
    """Raised when a requested tenant account is missing or suspended."""
    pass


class CollectionNotFoundError(CiteBaseError):
    """Raised when an operation targets a non-existent document collection."""
    pass


class IngestionError(CiteBaseError):
    """Raised when document ingestion, parsing, or indexing fails."""
    pass


class RateLimitExceededError(CiteBaseError):
    """Raised when a tenant exceeds their allotted request quota."""
    pass


class CacheError(CiteBaseError):
    """Raised when cache read or write operations encounter an unrecoverable failure."""
    pass
