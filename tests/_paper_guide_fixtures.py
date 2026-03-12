from __future__ import annotations

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
    return {
        "md_main": md_main,
        "blocks": blocks,
        "block_lookup": block_lookup,
        "wrong_method_block": wrong_method_block,
        "conclusion_block": conclusion_block,
        "eq3_block": eq3_block,
    }
