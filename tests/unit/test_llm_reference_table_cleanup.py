from types import SimpleNamespace

from kb.converter.llm_reference_table_cleanup import llm_polish_references


class _DummyResp:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _DummyWorker:
    def __init__(self, content: str):
        self._client = object()
        self._content = content

    def _llm_create(self, **kwargs):
        return _DummyResp(self._content)


class _DummyConverter:
    def __init__(self, content: str):
        self.cfg = SimpleNamespace(llm=SimpleNamespace(max_tokens=4096, api_key="k", base_url="u"))
        self.llm_worker = _DummyWorker(content)


def test_llm_polish_references_repairs_text_but_preserves_numbers():
    md = "\n".join(
        [
            "## References",
            "",
            "[1] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. CVPR, 2016.",
            "[2] Xin Miao, Xin Yuan. lnet: Reconstruct hyperspectral images. ICCV, 2019.",
            "[3] Haithem Turki et al. Mega-nerf: Scalable construction of largescale nerfs. CVPR, 2022.",
        ]
    )
    repaired = "\n".join(
        [
            "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. CVPR, 2016.",
            "[2] Xin Miao, Xin Yuan. I-net: Reconstruct hyperspectral images. ICCV, 2019.",
            "[3] Haithem Turki et al. Mega-NeRF: Scalable construction of large-scale NeRFs. CVPR, 2022.",
        ]
    )

    out = llm_polish_references(_DummyConverter(repaired), md)

    assert "Structure-from-motion revisited" in out
    assert "I-net: Reconstruct hyperspectral images" in out
    assert "large-scale NeRFs" in out
    assert "[1]" in out and "[2]" in out and "[3]" in out


def test_llm_polish_references_falls_back_when_numbering_changes():
    md = "\n".join(
        [
            "## References",
            "",
            "[1] A. First reference.",
            "[2] B. Second reference.",
            "[3] C. Third reference.",
        ]
    )
    bad = "\n".join(
        [
            "[1] A. First reference.",
            "[4] B. Second reference.",
        ]
    )

    out = llm_polish_references(_DummyConverter(bad), md)

    assert out == md
