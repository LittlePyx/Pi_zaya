from __future__ import annotations

from api.claim_passage_plan import _supports
from kb.evidence_binding import assess_system_a_hit_binding
from kb.explicit_noise_relations import named_synthetic_sample_covered, sampling_ability_covered
from kb.explicit_training_assignments import noise_augmentation_training_covered


def test_experiment_noise_levels_do_not_define_an_augmentation_training_recipe():
    claim = 'Noisier2Noise 研究的是在无配对噪声数据上训练去噪网络，其做法是向已有噪声图像再加入合成噪声，并让网络预测原噪声图像。'
    experiment = 'a fixed noise level during training. For DIP, we use the authors implementation and default hyperparameters. For Noise2Void we train on the Color BSD68 dataset. We obtain PSNR values on our test set.'
    source = 'Given Y, we create a noisier version, Z = Y + M. We then train the network to predict Y given Z as input.'
    assert not noise_augmentation_training_covered(claim, experiment)
    assert not _supports(claim, experiment, 'Experiments', 'noisier2noise.en.md')
    assert noise_augmentation_training_covered(claim, source)
    assert not noise_augmentation_training_covered(claim, 'We augment the inputs with additional noise and predict the clean image X.')
    bare = 'Noisier2Noise 讨论的是在已有噪声图像上叠加合成噪声、让网络预测原始噪声图像的去噪训练方式。'
    assert not _supports(bare, experiment, '', 'noisier2noise.en.md')


def test_psnr_alone_cannot_support_a_psnr_and_ssim_comparison():
    claim = 'The paper reports average PSNR and SSIM comparisons on KODAK under structured noise.'
    assert not _supports(claim, 'We obtain PSNR values on KODAK under structured noise.', 'Experiments', 'paper.pdf')
    assert _supports(claim, 'The paper reports average PSNR and SSIM comparisons on KODAK under structured noise.', '', 'paper.pdf')
    binding = assess_system_a_hit_binding(answer_claim=claim, hit={'text':claim}, meta={'citation_plan_evidence_authoritative':True,'citation_plan_full_evidence_quote':'We obtain PSNR values on KODAK under structured noise.'}, heading='PSNR and SSIM', evidence_quote='We obtain PSNR values on KODAK under structured noise.', source_name='paper.pdf')
    assert binding['suppress_link']


def test_sampling_requirement_cannot_become_pronoun_based_incapacity():
    claim = '本文的额外要求是能够从噪声分布中采样，这是前述三种方法所不具备的。'
    source = 'we require the ability to sample from the noise distribution, which the three aforementioned methods do not.'
    assert not sampling_ability_covered(claim, source)
    assert not _supports(claim, source, 'Related Work', 'paper.pdf')


def test_baseline_names_cannot_supply_a_qualitative_experimental_result():
    claim = 'Our method outperforms BM3D and TWSC on real-world noise.'
    assert not _supports(claim, 'We compare our method to BM3D and TWSC on real-world noise.', 'Experiments', 'paper.pdf')
    assert _supports(claim, 'Our method outperforms BM3D and TWSC on real-world noise.', '', 'paper.pdf')
    assert not _supports(claim, 'Our method does not outperform BM3D and TWSC on real-world noise.', '', 'paper.pdf')
    assert not _supports(claim, 'Our method is not better than BM3D and TWSC on real-world noise.', '', 'paper.pdf')
    assert not _supports(claim.replace('outperforms', 'does not outperform'), 'Our method outperforms BM3D and TWSC on real-world noise.', '', 'paper.pdf')
    assert _supports(claim, 'Our method not only outperforms BM3D and TWSC on real-world noise but is also faster.', '', 'paper.pdf')


def test_original_noise_variable_cannot_become_the_additional_synthetic_sample():
    claim = r'因为噪声分布已知，才能额外抽取一个合成噪声样本（$N \sim \mathcal{A}$）。'
    source = 'N is drawn from a known noise distribution. Because it is known, we can draw an additional synthetic noise sample.'
    assert not named_synthetic_sample_covered(claim, source)
    assert not _supports(claim, source, '', 'paper.pdf')
    assert not named_synthetic_sample_covered(claim, 'M is a synthetic sample from the same noise distribution.')
    assert named_synthetic_sample_covered(claim.replace('$N', '$M'), 'M is a synthetic sample from the same noise distribution.')
    assert not named_synthetic_sample_covered(claim.replace('$N', '$m'), 'M is a synthetic sample from the same noise distribution.')
