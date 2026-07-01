"""Send documents to Gemini as inline base64 bytes instead of the Files API.

Inspect's Google provider hardcodes document handling: images are sent as
inline `Part.from_bytes`, but any `ContentDocument` (e.g. our PDFs) is
uploaded via the Gemini Files API first. Importing this module patches the
provider so documents are always inlined like images (base64 `inlineData` in
the `generateContent` request), avoiding any Files API dependency (e.g. when
routing through a proxy that only supports `generateContent`).

NOTE: this patches a private Inspect module (`inspect_ai.model._providers
.google`, `chat_content_to_part`) and is sensitive to Inspect refactors;
verified against inspect-ai 0.3.244.
"""

from __future__ import annotations

import inspect_ai.model._providers.google as _google
from inspect_ai.model import ContentDocument

# Documents are always inlined, with no size cutoff: the dataset's largest
# PDFs (~33MB) exceed AI Studio's ~20MB inline request limit but are accepted
# by Vertex-backed endpoints, and an oversized request fails with a clear 400
# from the API rather than a confusing Files API error.
_original_chat_content_to_part = _google.chat_content_to_part


async def _chat_content_to_part_inline_documents(client, content):  # type: ignore[no-untyped-def]
    if isinstance(content, ContentDocument):
        data, mime_type = await _google.file_as_data(content.document)
        return _google.Part.from_bytes(mime_type=mime_type, data=data)
    return await _original_chat_content_to_part(client, content)


def install() -> None:
    """Apply the patch (idempotent). Called from `gdp_pdf.task`."""
    if _google.chat_content_to_part is not _chat_content_to_part_inline_documents:
        _google.chat_content_to_part = _chat_content_to_part_inline_documents
