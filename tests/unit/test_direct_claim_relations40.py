from __future__ import annotations

import pytest

from kb.explicit_training_assignments import noise_augmentation_training_covered
from kb.explicit_source_attribution import source_attribution_covered
from kb.explicit_noise_relations import signal_dependency_covered
from kb.theory_evidence import measurement_conditions_covered
from kb.explicit_claim_metrics import qualitative_outperformance_covered
from kb.citation_sentences import split_definition_bridges, citation_sentence_spans
from kb.waived_data_requirements import waived_data_requirements_covered
from kb.direct_question_evidence import direct_question_evidence_hits, prioritize_selected_source_relation_hits
from kb.explicit_noise_relations import sampling_ability_covered

CORRECTION='We can therefore recover an estimate of the clean image by doubling our network output and subtracting its input.'
DRAW='We consider a scenario in which we wish to train a neural network under a fixed, known noise model. We cannot observe X but observe Y = X + N. Because the distribution is known, we can draw an additional synthetic noise sample M.'
RECIPE='Our training procedure is as follows. Given Y, we create a noisier version, Z = Y + M = X + N + M. We then train the network to predict Y given Z as input.'

@pytest.mark.parametrize('claim',[
 'The network is trained to predict the singly-noisy image Y from the doubly-noisy input Z, not the clean image X.',
 'The network predicts the noisy observation Y given the doubly-noisy image Z.',
])
def test_descriptive_named_prediction_needs_the_actual_training_map(claim):
 assert not source_attribution_covered(claim,CORRECTION,'paper.pdf')
 assert source_attribution_covered(claim,RECIPE,'paper.pdf')

@pytest.mark.parametrize('claim',[
 'Noisier2Noise is the method that adds an additional synthetic noise sample drawn from a known noise distribution.',
 'The network adds synthetic noise to the observed noisy image.',
 'Noisier2Noise 额外加入合成噪声。',
])
def test_drawing_noise_is_not_the_operation_of_adding_it(claim):
 assert not noise_augmentation_training_covered(claim,DRAW)
 assert noise_augmentation_training_covered(claim,RECIPE)
 assert noise_augmentation_training_covered(claim,'We augment the noisy training images with an additional synthetic noise sample.')
 assert noise_augmentation_training_covered('We draw an additional synthetic noise sample M.',DRAW)

@pytest.mark.parametrize('claim',[
 'Noise2Self 的摘要写明噪声在不同测量维度之间表现出统计独立，而真实信号表现出一定相关性。',
 'This fits the general assumption that noise is statistically independent across measurement dimensions while the true signal exhibits some correlation.',
])
def test_modality_and_conditional_noise_passages_do_not_supply_signal_correlation(claim):
 actual='The only assumption is that the noise exhibits statistical independence across different dimensions of the measurement, while the true signal exhibits some correlation.'
 wrong='We use groups of features whose noise is independent conditional on the true signal to predict one another.'
 assert not signal_dependency_covered(claim,wrong)
 assert signal_dependency_covered(claim,actual)

def test_enabling_condition_is_separate_from_a_function_definition():
 claim='This property enables self-supervision when the noise is conditionally independent between sets of dimensions.'
 definition='J-invariance means f(x)J does not depend on xJ.'
 assert not measurement_conditions_covered(claim,definition)
 assert measurement_conditions_covered(claim,definition+' This enables self-supervision when the noise in the data is conditionally independent between sets of dimensions.')

def test_reported_moderation_is_not_in_a_data_requirement_alone():
 claim='论文自己指出，不能指望 N2V 超过训练时拥有更多信息的方法，其去噪性能只是适度下降。'
 waiver='It does not require noisy image pairs, nor clean target images.'
 actual='N2V cannot be expected to outperform methods that have more information available during training. Still, we observe that denoising performance drops in moderation.'
 assert not qualitative_outperformance_covered(claim,waiver)
 assert qualitative_outperformance_covered(claim,actual)

def test_distinct_observation_and_addition_predicates_split_without_rewriting():
 text='The procedure starts from a single noisy realization Y from a known noise distribution, and adds a synthetic noise sample M to form Z.'
 parts=split_definition_bridges(text)
 assert len(parts)==2
 assert ''.join(text[a:b] for a,b in citation_sentence_spans(text))==text
 for prefix in ('If ','It is false that '):
  assert len(split_definition_bridges(prefix+text))==1

def test_chinese_no_requirement_is_not_positive_clean_supervision():
 claim='Noise2Void 的训练方案明确不要求带噪图像对，也不要求干净目标图像，因此每个场景只保留一张带噪图、没有干净参照时，仍可直接在这批待去噪数据上训练。'
 source='We introduce Noise2Void. It does not require noisy image pairs, nor clean target images. Consequently, N2V allows us to train directly on the body of data to be denoised.'
 assert source_attribution_covered(claim,source,'noise2void.pdf')
 assert not source_attribution_covered(claim.replace('不要求干净目标图像','要求干净目标图像'),source,'noise2void.pdf')

def test_loss_gap_selection_keeps_a_correct_difference_term_paraphrase():
 claim='之所以能不用干净参照来选择去噪器，是因为这个差值项 $E\\|x-y\\|^2$ 与去噪器 $f$ 无关；'
 proof='That is, the self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise. By minimizing the self-supervised loss over a class of J-invariant functions, one may find the optimal denoiser for a given dataset.'
 assert waived_data_requirements_covered(claim,proof)
 assert not waived_data_requirements_covered(claim,DRAW)
 assert not waived_data_requirements_covered('因此无需干净训练数据即可训练去噪器。',proof)

def test_residual_training_loss_cannot_use_a_blind_spot_caption():
 claim='Its loss is the mean squared error between the desired residual images and the estimated ones, computed over noisy-clean training pairs.'
 wrong='The receptive field excludes the pixel itself, preventing it from learning the identity. We train on the same noisy images as input and target.'
 actual='The averaged mean squared error between the desired residual images and estimated ones can be adopted as the loss function in DnCNN. Here the pairs represent noisy-clean training images.'
 assert not source_attribution_covered(claim,wrong,'noise2void.pdf')
 assert source_attribution_covered(claim,actual,'dncnn.pdf')

@pytest.mark.parametrize('claim',[
 'Noise2Void 论文对 Noise2Noise 的概括是同一内容的两次独立噪声观测。',
 'Noise2Void 明确指出 Noise2Noise 需要同一内容的两次独立噪声观测。',
])
def test_paper_named_as_the_reporter_must_be_the_cited_paper(claim):
 source='We observe each source image twice with independent noises.'
 assert not source_attribution_covered(claim,source,'noise2noise.pdf')
 assert source_attribution_covered(claim,source,'noise2void.pdf')

def test_a_requirement_is_not_an_inability():
 claim='它要求能够从噪声分布中采样，而上述三种方法不具备这一能力。'
 source='We require the ability to sample from the noise distribution, which none of these methods require.'
 assert not sampling_ability_covered(claim,source)
 assert sampling_ability_covered('它要求能够从噪声分布中采样。',source)

def test_training_augmentation_keeps_both_premise_and_operation_in_context(tmp_path):
 paper=tmp_path/'noisier.md'
 paper.write_text('# Noisier method\n\n<!-- kb_page: 3 -->\n\n'+DRAW+'\n\n'+RECIPE,encoding='utf8')
 question="Compare Self2Self and Noisier2Noise training requirements using each paper's own method: which constructs Bernoulli-masked instances of the very image being denoised, and which adds sampled noise from a statistical noise model?"
 hits=direct_question_evidence_hits(paper,question,limit=6)
 assert len(hits)==2 and hits[0]['text']==RECIPE
 other={'text':'Another selected paper remains selected.','meta':{'source_path':str(tmp_path/'self.md')}}
 selected=prioritize_selected_source_relation_hits([hits[1],other],hits,limit=6,prompt=question)
 assert len(selected)==2 and selected[1]==other
 assert DRAW in selected[0]['text'] and RECIPE in selected[0]['text']
 assert len(selected[0]['meta']['source_passages'])==2
 assert {p['block_id'] for p in selected[0]['meta']['source_passages']}=={h['meta']['block_id'] for h in hits}

def test_j_invariance_request_retrieves_the_identity_exclusion_as_well(tmp_path):
 paper=tmp_path/'self.md'
 definition='J is a subset of the dimensions, and f is a J-invariant function: f(x)J does not depend on xJ.'
 reason='We minimize over J-invariant functions f. Since f has to use information from outside of each subset of dimensions J to predict the values inside of J, it cannot merely be the identity.'
 paper.write_text('# Denoising\n\n<!-- kb_page: 2 -->\n\n'+definition+'\n\n'+reason,encoding='utf8')
 hits=direct_question_evidence_hits(paper,'What does J-invariance require of the output on a coordinate subset J, and why does that rule prevent an identity predictor?')
 assert hits and definition in hits[0]['text']
 assert any(reason in h['text'] for h in hits)

def test_stable_training_needs_more_than_a_speed_statement():
 claim='Their integration yields fast and stable training with better denoising performance.'
 speed='Residual learning and batch normalization benefit from each other, and their integration speeds up training and boosts denoising performance.'
 assert not source_attribution_covered(claim,speed,'dncnn.pdf')
 assert source_attribution_covered(claim,speed+' The integration results in fast and stable training.','dncnn.pdf')
