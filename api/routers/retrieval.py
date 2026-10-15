from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import get_settings
from api.internal_access import require_management_api
from kb.local_semantic_preparation_service import (
    LOCAL_SEMANTIC_PRODUCT_ENABLED,
    LocalSemanticPreparationServiceError,
    get_local_semantic_preparation_service,
)
from kb.local_semantic_component_lifecycle import (
    LocalSemanticComponentLifecycleError,
)
from kb.local_semantic_component_service import (
    LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED,
    get_local_semantic_component_service,
    local_semantic_component_disclosure,
    local_semantic_component_management_available,
)


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.get("/local-readiness")
def local_retrieval_readiness() -> dict[str, object]:
    readiness = get_local_semantic_preparation_service(get_settings().db_dir).status()
    available_modes = ["bm25"]
    if readiness.available and LOCAL_SEMANTIC_PRODUCT_ENABLED:
        available_modes.append("hybrid_local")
    return {
        "schema_version": 1,
        "default_mode": "bm25",
        "available_modes": available_modes,
        "hybrid_local": readiness.public_payload(),
        "show_mode_control": readiness.available and LOCAL_SEMANTIC_PRODUCT_ENABLED,
    }


class LocalPreparationBody(BaseModel):
    action: str


class LocalPreparationCancelBody(BaseModel):
    task_id: str


class LocalComponentEnableBody(BaseModel):
    consent: bool = False


class LocalComponentCancelBody(BaseModel):
    task_id: str


def _local_component_payload(
    status,
    *,
    created: bool | None = None,
) -> dict[str, object]:
    install_available = local_semantic_component_management_available()
    product_available = bool(
        LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED and status.available
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "default_mode": "bm25",
        "available_modes": ["bm25", "hybrid_local"] if product_available else ["bm25"],
        "show_mode_control": bool(
            LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED and install_available
        ),
        "product_enabled": LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED,
        "install_available": install_available,
        "component": local_semantic_component_disclosure(),
        "hybrid_local": status.public_payload(),
    }
    if created is not None:
        payload["created"] = bool(created)
    return payload


def _raise_component_error(exc: LocalSemanticComponentLifecycleError) -> None:
    raise HTTPException(status_code=409, detail=exc.code) from exc


@router.get(
    "/local-component",
    dependencies=[Depends(require_management_api)],
)
def local_semantic_component_status() -> dict[str, object]:
    service = get_local_semantic_component_service(get_settings().db_dir)
    return _local_component_payload(service.status())


@router.post(
    "/local-component/enable",
    dependencies=[Depends(require_management_api)],
)
def enable_local_semantic_component(
    body: LocalComponentEnableBody,
) -> dict[str, object]:
    service = get_local_semantic_component_service(get_settings().db_dir)
    try:
        status, created = service.enable(consent=body.consent)
    except LocalSemanticComponentLifecycleError as exc:
        _raise_component_error(exc)
    return _local_component_payload(status, created=created)


@router.post(
    "/local-component/resume",
    dependencies=[Depends(require_management_api)],
)
def resume_local_semantic_component() -> dict[str, object]:
    service = get_local_semantic_component_service(get_settings().db_dir)
    try:
        status, created = service.resume()
    except LocalSemanticComponentLifecycleError as exc:
        _raise_component_error(exc)
    return _local_component_payload(status, created=created)


@router.post(
    "/local-component/cancel",
    dependencies=[Depends(require_management_api)],
)
def cancel_local_semantic_component(
    body: LocalComponentCancelBody,
) -> dict[str, object]:
    service = get_local_semantic_component_service(get_settings().db_dir)
    try:
        status = service.cancel(body.task_id)
    except LocalSemanticComponentLifecycleError as exc:
        _raise_component_error(exc)
    return _local_component_payload(status)


@router.delete(
    "/local-component",
    dependencies=[Depends(require_management_api)],
)
def remove_local_semantic_component() -> dict[str, object]:
    service = get_local_semantic_component_service(get_settings().db_dir)
    try:
        status, created = service.remove()
    except LocalSemanticComponentLifecycleError as exc:
        _raise_component_error(exc)
    return _local_component_payload(status, created=created)


@router.post(
    "/local-preparation",
    dependencies=[Depends(require_management_api)],
)
def start_local_retrieval_preparation(body: LocalPreparationBody) -> dict[str, object]:
    service = get_local_semantic_preparation_service(get_settings().db_dir)
    try:
        status, created = service.start(body.action)
    except LocalSemanticPreparationServiceError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from exc
    return {"schema_version": 1, "created": created, "hybrid_local": status.public_payload()}


@router.post(
    "/local-preparation/cancel",
    dependencies=[Depends(require_management_api)],
)
def cancel_local_retrieval_preparation(
    body: LocalPreparationCancelBody,
) -> dict[str, object]:
    service = get_local_semantic_preparation_service(get_settings().db_dir)
    try:
        status = service.cancel(body.task_id)
    except LocalSemanticPreparationServiceError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from exc
    return {"schema_version": 1, "hybrid_local": status.public_payload()}
