from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports

TABLE = ('1%: PNSR (dB) / Hadamard / circular = 8.01; '
         'PNSR (dB) / Fourier / circular = 8.08; '
         'SSIM (%) / Hadamard / circular = 10.0; SSIM (%) / Fourier / circular = 11.1')
CLAIM = '欠采样时，Fourier 单像素成像（FSI）的重建质量优于 Hadamard（HSI）。'


def test_table_support_requires_the_named_comparison_direction():
    assert _supports(CLAIM, TABLE, '', 'Hadamard versus Fourier')
    assert not _supports(CLAIM.replace('Fourier', 'TEMP').replace('Hadamard', 'Fourier').replace('TEMP', 'Hadamard'), TABLE, '', 'Hadamard versus Fourier')


@pytest.mark.parametrize('table', [TABLE.replace('8.08', '7.08'), TABLE.replace('11.1', '9.1'), TABLE.replace('8.08', '8.01'), TABLE.replace('Fourier / circular', 'Fourier / different'), TABLE + '; PNSR (dB) / Fourier / circular = 7.08'])
def test_conflicting_metrics_conditions_ties_or_duplicate_values_cannot_prove_superiority(table):
    assert not _supports(CLAIM, table, '', 'Hadamard versus Fourier')


def test_numeric_own_method_comparison_keeps_both_comparators_and_their_direction():
    claim = '该方法的 SSIM 达到 0.76，优于 VST+bicubic 的 0.50 和 U-net 的 0.60。'
    evidence = 'Table 2 shows SSIM results: VST+bicubic = 0.50, U-net = 0.60, Ours = 0.76.'
    assert _supports(claim, evidence, '', 'paper.pdf')
    assert not _supports(claim.replace('0.76', '0.56'), evidence.replace('0.76', '0.56'), '', 'paper.pdf')
