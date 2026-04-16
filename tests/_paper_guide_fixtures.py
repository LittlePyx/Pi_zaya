from __future__ import annotations

import json
from pathlib import Path


def build_scinerf_like_fixture(tmp_path: Path) -> dict[str, object]:
    from kb import task_runtime

    md_dir = tmp_path / "SCINeRF"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "SCINeRF.en.md"
    md_main.write_text(
        (
            "# Method\n\n"
            "We render $x_i$ to synthesize the compressed image $y$ for the SCI observation model.\n\n"
            "# Background on NeRF\n\n"
            "$$Y = \\sum_{i=1}^{N} X_i \\odot M_i + Z \\tag{3}$$\n\n"
            "# Discussion\n\n"
            "Looking ahead, a practical next step is to extend the method to dynamic scenes while keeping the storage and hardware burden low through software-based SCI implementations.\n\n"
            "# Limitations\n\n"
            "A current limitation is that the method trades temporal coverage against reconstruction stability when the captured scene deviates too far from the static-scene assumption.\n\n"
            "# Future Work\n\n"
            "A direct future extension is to combine the method with adaptive masking so dynamic scenes can be reconstructed more faithfully without increasing the hardware budget.\n\n"
            "# Conclusion\n\n"
            "SCINeRF exploits neural radiance fields as its underlying scene representation. "
            "Physical image formation process of an SCI image is exploited to formulate the training objective "
            "for jointly NeRF training and camera poses optimization.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    block_lookup = {
        str(block.get("block_id") or "").strip(): dict(block)
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "").strip()
    }
    wrong_method_block = next(
        block
        for block in blocks
        if "render $x_i$ to synthesize the compressed image $y$" in str(block.get("text") or "").lower()
    )
    conclusion_block = next(
        block
        for block in blocks
        if "scinerf exploits neural radiance fields as its underlying scene representation"
        in str(block.get("text") or "").lower()
    )
    eq3_block = next(
        block
        for block in blocks
        if str(block.get("kind") or "") == "equation" and int(block.get("number") or 0) == 3
    )
    discussion_block = next(
        block
        for block in blocks
        if "extend the method to dynamic scenes" in str(block.get("text") or "").lower()
    )
    limitations_block = next(
        block
        for block in blocks
        if "trades temporal coverage against reconstruction stability" in str(block.get("text") or "").lower()
    )
    future_work_block = next(
        block
        for block in blocks
        if "combine the method with adaptive masking" in str(block.get("text") or "").lower()
    )
    return {
        "md_main": md_main,
        "blocks": blocks,
        "block_lookup": block_lookup,
        "wrong_method_block": wrong_method_block,
        "conclusion_block": conclusion_block,
        "eq3_block": eq3_block,
        "discussion_block": discussion_block,
        "limitations_block": limitations_block,
        "future_work_block": future_work_block,
    }


def build_paper_guide_runtime_fixture(tmp_path: Path) -> dict[str, object]:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    nat_stem = "NatPhoton-2019-SPI"
    nat_dir = db_root / nat_stem
    nat_dir.mkdir(parents=True, exist_ok=True)
    nat_md = nat_dir / f"{nat_stem}.en.md"
    nat_md.write_text(
        (
            "## Acquisition and image reconstruction strategies\n\n"
            "<!-- box:start id=1 -->\n"
            "**[Box 1 - The maths behind single-pixel imaging]**\n\n"
            "It can be shown that when the number of sampling patterns used "
            "$M \\ge O(K \\log(N/K))$, the image in the transform domain can be reconstructed.\n"
            "<!-- box:end id=1 -->\n\n"
            "An alternative approach is to perform sampling using a basis that is not necessarily incoherent\n"
            "with the spatial properties of the image, for example by using the Hadamard$^{64,65}$ basis.\n"
        ),
        encoding="utf-8",
    )

    lsa_stem = "LSA-2026-iISM-live-cells"
    lsa_dir = db_root / lsa_stem
    lsa_dir.mkdir(parents=True, exist_ok=True)
    lsa_md = lsa_dir / f"{lsa_stem}.en.md"
    lsa_md.write_text(
        (
            "## Results / APR\n\n"
            "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram\n"
            "into an intensity-only map.\n"
        ),
        encoding="utf-8",
    )

    lsa_md_abs = lsa_md.resolve(strict=False)
    refs_doc = {
        "path": str(lsa_md_abs),
        "name": lsa_md.name,
        "stem": lsa_md.stem.lower(),
        "sha1": "",
        "refs": {
            "34": {
                "num": 34,
                "raw": "[34] Precision single-particle localization using radial variance transform.",
                "title": "Precision single-particle localization using radial variance transform",
            }
        },
    }
    refs_index = {
        "version": 1,
        "updated_at": 0.0,
        "doc_count": 1,
        "next_cursor": 1,
        "docs": {
            str(lsa_md_abs).strip().lower(): refs_doc,
        },
    }
    (db_root / "references_index.json").write_text(
        json.dumps(refs_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "db_root": db_root,
        "nat_source_path": rf"X:\{nat_stem}.pdf",
        "lsa_source_path": rf"X:\{lsa_stem}.pdf",
        "nat_md": nat_md,
        "lsa_md": lsa_md,
    }
