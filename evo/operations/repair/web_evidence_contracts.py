from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
WebUrl = Annotated[StrictStr, Field(min_length=1, max_length=2048)]
SearchQuery = Annotated[StrictStr, Field(min_length=1, max_length=500)]
WebEvidenceStatus = Literal['ready', 'partial', 'unavailable']
WebEvidenceWarningCode = Literal[
    'invalid_search_result',
    'duplicate_search_result',
    'search_result_limit_reached',
    'search_unavailable',
    'no_search_results',
    'no_page_selected',
    'warning_limit_reached',
    'fetch_failed',
    'fetch_result_missing',
    'unsupported_content_type',
    'empty_page_content',
    'duplicate_page_content',
    'content_truncated',
    'total_content_budget_exhausted',
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class CleanSearchResult(StrictModel):
    """Provider-independent search result retained by Repair."""

    title: StrictStr = Field(
        min_length=1,
        max_length=300,
        description='Title retained from the corresponding web_search result.',
    )
    url: WebUrl
    snippet: StrictStr = Field(max_length=1000)

    @field_validator('title', mode='before')
    @classmethod
    def normalize_title(cls, value: Any) -> Any:
        return _normalize_required_text(value, 'search result title')

    @field_validator('snippet', mode='before')
    @classmethod
    def normalize_snippet(cls, value: Any) -> Any:
        return _normalize_optional_text(value)

    @field_validator('url', mode='before')
    @classmethod
    def normalize_url(cls, value: Any) -> Any:
        return _normalize_web_url(value)


class WebEvidenceWarning(StrictModel):
    code: WebEvidenceWarningCode
    message: StrictStr = Field(min_length=1, max_length=500)
    url: WebUrl | None = None
    queries: list[SearchQuery] = Field(default_factory=list, max_length=32)

    @field_validator('message', mode='before')
    @classmethod
    def normalize_message(cls, value: Any) -> Any:
        return _normalize_required_text(value, 'warning message')

    @field_validator('url', mode='before')
    @classmethod
    def normalize_url(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_web_url(value)

    @field_validator('queries', mode='before')
    @classmethod
    def normalize_queries(cls, value: Any) -> Any:
        return _normalize_text_list(value, 'warning query')

    @model_validator(mode='after')
    def validate_unique_queries(self) -> Self:
        _require_unique(self.queries, 'warning query')
        return self


class SearchResultSet(StrictModel):
    """Normalized results for exactly one search query."""

    id: Literal['repair.web_search_results'] = 'repair.web_search_results'
    schema_version: Literal['1'] = '1'
    query: SearchQuery
    results: list[CleanSearchResult] = Field(default_factory=list, max_length=50)
    warnings: list[WebEvidenceWarning] = Field(default_factory=list, max_length=100)

    @field_validator('query', mode='before')
    @classmethod
    def normalize_query(cls, value: Any) -> Any:
        return _normalize_required_text(value, 'search query')

    @model_validator(mode='after')
    def validate_unique_urls(self) -> Self:
        _require_unique((item.url for item in self.results), 'search result URL')
        return self


class PageSelectionItem(StrictModel):
    """One candidate URL selected from normalized search results.

    Membership of ``url`` in the corresponding normalized search results is a
    business-layer check because this standalone contract does not own those
    result sets.
    """

    url: WebUrl

    @field_validator('url', mode='before')
    @classmethod
    def normalize_url(cls, value: Any) -> Any:
        return _normalize_web_url(value)


class PageSelection(StrictModel):
    id: Literal['repair.web_page_selection'] = 'repair.web_page_selection'
    schema_version: Literal['1'] = '1'
    selected_pages: list[PageSelectionItem] = Field(min_length=1, max_length=3)

    @model_validator(mode='after')
    def validate_unique_urls(self) -> Self:
        _require_unique(
            (item.url for item in self.selected_pages),
            'selected page URL',
        )
        return self


class WebEvidencePage(StrictModel):
    """Cleaned, exact page excerpts retained as external Repair evidence."""

    evidence_id: Identifier
    queries: list[SearchQuery] = Field(min_length=1, max_length=32)
    title: StrictStr = Field(
        min_length=1,
        max_length=300,
        description='Title retained from the corresponding web_search result.',
    )
    url: WebUrl
    final_url: WebUrl
    snippet: StrictStr = Field(max_length=1000)
    content: StrictStr = Field(min_length=1, max_length=50_000)
    content_sha256: StrictStr = Field(pattern=r'^[0-9a-f]{64}$')
    character_count: StrictInt = Field(ge=1, le=50_000)
    truncated: StrictBool

    @field_validator('evidence_id', 'title', mode='before')
    @classmethod
    def normalize_required_text(cls, value: Any, info: Any) -> Any:
        return _normalize_required_text(value, info.field_name)

    @field_validator('queries', mode='before')
    @classmethod
    def normalize_queries(cls, value: Any) -> Any:
        return _normalize_text_list(value, 'search query')

    @field_validator('url', 'final_url', mode='before')
    @classmethod
    def normalize_urls(cls, value: Any) -> Any:
        return _normalize_web_url(value)

    @field_validator('snippet', mode='before')
    @classmethod
    def normalize_snippet(cls, value: Any) -> Any:
        return _normalize_optional_text(value)

    @field_validator('content')
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('content must contain non-whitespace text')
        return value

    @model_validator(mode='after')
    def validate_content_metadata(self) -> Self:
        _require_unique(self.queries, 'web evidence query')
        if self.character_count != len(self.content):
            raise ValueError(
                'character_count must equal the number of characters in content'
            )
        expected_hash = sha256(self.content.encode('utf-8')).hexdigest()
        if self.content_sha256 != expected_hash:
            raise ValueError('content_sha256 must match the UTF-8 content')
        return self


class RepairWebEvidence(StrictModel):
    id: Literal['repair.web_evidence'] = 'repair.web_evidence'
    schema_version: Literal['1'] = '1'
    status: WebEvidenceStatus
    content_trust: Literal['external_untrusted'] = 'external_untrusted'
    pages: list[WebEvidencePage] = Field(default_factory=list, max_length=3)
    total_character_count: StrictInt = Field(ge=0, le=50_000)
    warnings: list[WebEvidenceWarning] = Field(default_factory=list, max_length=100)

    @model_validator(mode='after')
    def validate_result(self) -> Self:
        _require_unique(
            (page.evidence_id for page in self.pages),
            'web evidence id',
        )
        _require_unique((page.url for page in self.pages), 'web evidence URL')

        expected_character_count = sum(
            page.character_count
            for page in self.pages
        )
        if self.total_character_count != expected_character_count:
            raise ValueError(
                'total_character_count must equal the sum of page character counts'
            )

        if self.status == 'ready':
            if not self.pages:
                raise ValueError('ready web evidence requires at least one page')
        elif self.status == 'partial':
            if not self.pages:
                raise ValueError('partial web evidence requires at least one page')
            if not self.warnings:
                raise ValueError('partial web evidence requires at least one warning')
        else:
            if self.pages:
                raise ValueError('unavailable web evidence cannot contain pages')
            if not self.warnings:
                raise ValueError('unavailable web evidence requires at least one warning')

        truncated_urls = {
            page.url
            for page in self.pages
            if page.truncated
        }
        warned_urls = {
            warning.url
            for warning in self.warnings
            if warning.code == 'content_truncated' and warning.url is not None
        }
        missing_warnings = sorted(truncated_urls - warned_urls)
        if missing_warnings:
            raise ValueError(
                'truncated pages require content_truncated warnings: '
                f'{", ".join(missing_warnings)}'
            )
        return self


def _normalize_required_text(value: Any, label: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        raise ValueError(f'{label} must contain non-whitespace text')
    return text


def _normalize_optional_text(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _normalize_text_list(value: Any, label: str) -> Any:
    if not isinstance(value, list):
        return value
    result = []
    for item in value:
        result.append(_normalize_required_text(item, label))
    return result


def _normalize_web_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    url = value.strip()
    if not url:
        raise ValueError('web URL must contain non-whitespace text')
    if any(character.isspace() for character in url):
        raise ValueError('web URL cannot contain whitespace')
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError('web URL is invalid') from exc
    if parsed.scheme.lower() not in ('http', 'https'):
        raise ValueError('web URL scheme must be http or https')
    if not parsed.hostname:
        raise ValueError('web URL host is required')
    if parsed.username is not None or parsed.password is not None:
        raise ValueError('web URL credentials are not allowed')
    return url


def _require_unique(values: Any, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f'duplicate {label}: {", ".join(sorted(duplicates))}')


__all__ = [
    'CleanSearchResult',
    'PageSelection',
    'PageSelectionItem',
    'RepairWebEvidence',
    'SearchResultSet',
    'WebEvidencePage',
    'WebEvidenceStatus',
    'WebEvidenceWarning',
    'WebEvidenceWarningCode',
]
