"""Paper Guide contracts (compat import path).

The canonical contract implementation currently lives in :mod:`kb.paper_guide_contracts`.
This module provides a stable ``kb.paper_guide.contracts`` import path so that
new code can follow the package-oriented layout described in the docs without
forcing a disruptive move/rename of the existing module yet.
"""

from __future__ import annotations

from .. import paper_guide_contracts as _legacy

# Re-export common contract types for type checkers and IDEs.
PaperGuideIntentModel = _legacy.PaperGuideIntentModel
PaperGuideSupportRecordModel = _legacy.PaperGuideSupportRecordModel
PaperGuideSupportPackModel = _legacy.PaperGuideSupportPackModel
PaperGuideEvidenceCardModel = _legacy.PaperGuideEvidenceCardModel
PaperGuideRetrievalBundleModel = _legacy.PaperGuideRetrievalBundleModel
PaperGuideGroundingTraceSegmentModel = _legacy.PaperGuideGroundingTraceSegmentModel
PaperGuideCitationDetailModel = _legacy.PaperGuideCitationDetailModel
PaperGuideRenderPacketModel = _legacy.PaperGuideRenderPacketModel

PaperGuideRefSpan = _legacy.PaperGuideRefSpan
PaperGuideTargetScope = _legacy.PaperGuideTargetScope
PaperGuideEvidenceAtom = _legacy.PaperGuideEvidenceAtom
PaperGuideSupportSlot = _legacy.PaperGuideSupportSlot
PaperGuideSupportResolution = _legacy.PaperGuideSupportResolution


def __getattr__(name: str):  # pragma: no cover - thin re-export shim
    return getattr(_legacy, name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(dir(_legacy)))


__all__ = [
    "PaperGuideIntentModel",
    "PaperGuideSupportRecordModel",
    "PaperGuideSupportPackModel",
    "PaperGuideEvidenceCardModel",
    "PaperGuideRetrievalBundleModel",
    "PaperGuideGroundingTraceSegmentModel",
    "PaperGuideCitationDetailModel",
    "PaperGuideRenderPacketModel",
    "PaperGuideRefSpan",
    "PaperGuideTargetScope",
    "PaperGuideEvidenceAtom",
    "PaperGuideSupportSlot",
    "PaperGuideSupportResolution",
]

