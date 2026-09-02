from __future__ import annotations

import html
import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import requests

from kb.citation_meta import extract_first_doi


_DOI_HOSTS = {"doi.org", "dx.doi.org"}
_ARXIV_RE = re.compile(
    r"(?:arxiv\s*[:\s]|arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv[.:])(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_HTML_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_TITLE_RE = re.compile(
    r"<meta\b[^>]*(?:name|property)=[\"'](?:citation_title|og:title|twitter:title)[\"'][^>]*content=[\"']([^\"']+)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = {
    "about", "after", "also", "among", "article", "before", "from", "into", "paper",
    "results", "study", "that", "their", "these", "this", "through", "using", "with",
}
_MAX_VERIFY_BYTES = 256 * 1024
_MAX_REDIRECTS = 4


class ExternalSourceError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code or "external_source_error")
        self.detail = str(detail or "")
        super().__init__(self.detail or self.code)


def safe_external_url(value: object) -> str:
    raw = str(value or "").replace("\x00", "").strip()
    if not raw or len(raw) > 2400:
        return ""
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except Exception:
        return ""
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    if port not in {None, 443}:
        return ""
    host = str(parsed.hostname or "").strip("[]").lower().split("%", 1)[0]
    if not host or host == "localhost" or host.endswith((".localhost", ".local")):
        return ""
    netloc = host
    if ":" in host:
        netloc = f"[{host}]"
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


def _host_is_public(hostname: str) -> bool:
    host = str(hostname or "").strip("[]").lower().split("%", 1)[0]
    if not host or host == "localhost" or host.endswith((".localhost", ".local")):
        return False
    try:
        direct = ipaddress.ip_address(host)
        addresses = [direct]
    except ValueError:
        try:
            rows = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        except OSError:
            return False
        addresses = []
        for row in rows:
            try:
                addresses.append(ipaddress.ip_address(str(row[4][0]).split("%", 1)[0]))
            except (ValueError, IndexError):
                return False
    if not addresses:
        return False
    return all(
        not (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_unspecified
            or address.is_reserved
            or address.is_multicast
        )
        for address in addresses
    )


def _validated_external_url(value: object) -> str:
    url = safe_external_url(value)
    if not url:
        raise ExternalSourceError("unsafe_url", "Only public HTTPS sources are supported.")
    host = str(urlsplit(url).hostname or "")
    if not _host_is_public(host):
        raise ExternalSourceError("unsafe_host", "The source host is local, private, or cannot be resolved safely.")
    return url


def _response_headers(response: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in dict(getattr(response, "headers", {}) or {}).items()}


def _request_external(
    url: str,
    *,
    max_bytes: int,
    timeout_s: float,
    getter: Callable[..., Any] | None = None,
    truncate: bool = False,
) -> dict[str, Any]:
    current = _validated_external_url(url)
    get = getter or requests.get
    for redirect_count in range(_MAX_REDIRECTS + 1):
        try:
            response = get(
                current,
                headers={
                    "Accept": "application/pdf,text/html,application/xhtml+xml;q=0.9,*/*;q=0.3",
                    "User-Agent": "Pi-zaya-KB/0.1 (external source check)",
                },
                stream=True,
                allow_redirects=False,
                timeout=(min(2.0, timeout_s), timeout_s),
            )
        except requests.RequestException as exc:
            raise ExternalSourceError("unreachable", str(exc)[:240]) from exc
        except Exception as exc:
            raise ExternalSourceError("unreachable", str(exc)[:240]) from exc

        status = int(getattr(response, "status_code", 0) or 0)
        headers = _response_headers(response)
        if status in _REDIRECT_CODES:
            location = str(headers.get("location") or "").strip()
            try:
                response.close()
            except Exception:
                pass
            if not location or redirect_count >= _MAX_REDIRECTS:
                raise ExternalSourceError("redirect_failed", "The source redirect chain could not be completed safely.")
            current = _validated_external_url(urljoin(current, location))
            continue
        if status < 200 or status >= 300:
            try:
                response.close()
            except Exception:
                pass
            raise ExternalSourceError("http_error", f"Source returned HTTP {status}.")

        content_length = str(headers.get("content-length") or "").strip()
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = 0
            if declared_size > int(max_bytes) and not truncate:
                try:
                    response.close()
                except Exception:
                    pass
                raise ExternalSourceError("too_large", "The source exceeds the allowed size.")

        body = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > max_bytes:
                    if truncate:
                        del body[max_bytes:]
                        break
                    raise ExternalSourceError("too_large", "The source exceeds the allowed size.")
        finally:
            try:
                response.close()
            except Exception:
                pass
        return {
            "url": current,
            "status": status,
            "content_type": str(headers.get("content-type") or "").split(";", 1)[0].strip().lower(),
            "content_disposition": str(headers.get("content-disposition") or ""),
            "body": bytes(body),
        }
    raise ExternalSourceError("redirect_failed", "The source redirect chain is too long.")


def _decode_html(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _page_title(text: str) -> str:
    match = _META_TITLE_RE.search(text) or _HTML_TITLE_RE.search(text)
    if not match:
        return ""
    raw = re.sub(r"<[^>]+>", " ", str(match.group(1) or ""))
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()[:500]


def _title_matches(expected: str, actual: str) -> bool | None:
    left = str(expected or "").strip().lower()
    right = str(actual or "").strip().lower()
    if not left or not right:
        return None
    if left in right or right in left:
        return True
    left_tokens = set(_TOKEN_RE.findall(left)) - _STOPWORDS
    right_tokens = set(_TOKEN_RE.findall(right)) - _STOPWORDS
    if not left_tokens or not right_tokens:
        return None
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.34


def _claim_excerpt(answer: str, start_value: object, end_value: object) -> str:
    text = str(answer or "")
    try:
        start = int(start_value)
        end = int(end_value)
    except (TypeError, ValueError):
        return ""
    if start < 0 or start >= len(text) or end <= start or end > len(text):
        return ""
    left = max(0, start - 180)
    right = min(len(text), end + 40)
    prefix = text[left:start]
    boundary = max(prefix.rfind("。"), prefix.rfind("！"), prefix.rfind("？"), prefix.rfind("."), prefix.rfind("\n"))
    if boundary >= 0:
        left += boundary + 1
    excerpt = re.sub(r"\s+", " ", text[left:right]).strip()
    return excerpt[:320]


def _scholarly_metadata(url: str, title: str, page_text: str = "") -> dict[str, str]:
    haystack = "\n".join((unquote(url), title, page_text[:64 * 1024]))
    arxiv_match = _ARXIV_RE.search(haystack)
    arxiv_id = str(arxiv_match.group(1) or "").strip() if arxiv_match else ""
    doi = extract_first_doi(haystack)
    host = str(urlsplit(url).hostname or "").lower()
    if not doi and host in _DOI_HOSTS:
        doi = unquote(urlsplit(url).path).strip("/")
        doi = extract_first_doi(doi)
    out: dict[str, str] = {}
    if arxiv_id:
        out.update({
            "scholarly_kind": "arxiv",
            "arxiv_id": arxiv_id,
            "doi": doi or f"10.48550/arXiv.{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        })
    elif doi:
        out.update({"scholarly_kind": "doi", "doi": doi})
    else:
        out["scholarly_kind"] = "web"
    return out


def _verify_one(answer: str, citation: dict[str, Any], getter: Callable[..., Any] | None) -> dict[str, Any]:
    requested_url = safe_external_url(citation.get("url"))
    if not requested_url:
        raise ExternalSourceError("unsafe_url")
    result = _request_external(
        requested_url,
        max_bytes=_MAX_VERIFY_BYTES,
        timeout_s=3.0,
        getter=getter,
        truncate=True,
    )
    body = bytes(result.get("body") or b"")
    content_type = str(result.get("content_type") or "")
    is_pdf = content_type == "application/pdf" or body.startswith(b"%PDF-")
    page_text = "" if is_pdf else _decode_html(body)
    actual_title = _page_title(page_text)
    expected_title = str(citation.get("title") or "").replace("\x00", " ").strip()[:500]
    title_match = True if is_pdf else _title_matches(expected_title, actual_title)
    if title_match is False:
        raise ExternalSourceError("content_mismatch", "The returned page title does not match the cited source.")

    final_url = str(result.get("url") or requested_url)
    metadata = _scholarly_metadata(final_url, expected_title or actual_title, page_text)
    if is_pdf:
        metadata["pdf_url"] = final_url
    claim_text = _claim_excerpt(answer, citation.get("start_index"), citation.get("end_index"))
    source = {
        "title": expected_title or actual_title,
        "url": final_url,
        "domain": str(urlsplit(final_url).hostname or "").lower(),
        "verification_status": "verified",
        "verification_level": "content_match" if title_match is True else "reachable",
        "http_status": int(result.get("status") or 0),
        "content_type": content_type,
        "claim_text": claim_text,
        "start_index": citation.get("start_index"),
        "end_index": citation.get("end_index"),
        **metadata,
    }
    source["import_eligible"] = bool(source.get("pdf_url"))
    return source


def verify_web_citations(
    answer: str,
    citations: list[dict[str, Any]] | None,
    *,
    max_sources: int = 8,
    getter: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(citations or []):
        if not isinstance(raw, dict):
            continue
        url = safe_external_url(raw.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append({**raw, "url": url})
        if len(candidates) >= max(1, min(8, int(max_sources or 8))):
            break

    verified: list[tuple[int, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    if candidates:
        workers = min(4, len(candidates))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="web-source-check") as pool:
            futures = {
                pool.submit(_verify_one, str(answer or ""), item, getter): (index, item)
                for index, item in enumerate(candidates)
            }
            for future in as_completed(futures):
                index, item = futures[future]
                try:
                    verified.append((index, future.result()))
                except ExternalSourceError as exc:
                    rejected.append({"url": str(item.get("url") or ""), "reason": exc.code})
                except Exception:
                    rejected.append({"url": str(item.get("url") or ""), "reason": "verification_failed"})
    verified.sort(key=lambda row: row[0])
    return {
        "sources": [source for _, source in verified],
        "checked_count": len(candidates),
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def fetch_external_pdf(
    url: str,
    *,
    max_bytes: int,
    getter: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    result = _request_external(
        url,
        max_bytes=max(1, int(max_bytes)),
        timeout_s=15.0,
        getter=getter,
        truncate=False,
    )
    data = bytes(result.get("body") or b"")
    content_type = str(result.get("content_type") or "")
    if not data.startswith(b"%PDF-"):
        raise ExternalSourceError("not_pdf", "The selected source does not provide a direct PDF file.")
    final_url = str(result.get("url") or url)
    path_name = unquote(PurePosixPath(urlsplit(final_url).path).name).strip()
    file_name = path_name if path_name.lower().endswith(".pdf") else "external-source.pdf"
    return {
        "data": data,
        "url": final_url,
        "content_type": content_type or "application/pdf",
        "file_name": file_name,
    }


__all__ = [
    "ExternalSourceError",
    "fetch_external_pdf",
    "safe_external_url",
    "verify_web_citations",
]
