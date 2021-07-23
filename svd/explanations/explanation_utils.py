from audioLIME.factorization_slime import TimeFrequencyTorchFactorization
from audioLIME.lime_audio import LimeAudioExplainer
from svd.utils.config import parse_config_file
from torchaudio.transforms import Spectrogram
from svd.train import prep_net_config
from svd.utils.io import load_model
from svd.core.model import SingingVoiceDetector
from svd.path_config import default_vars_path
import numpy as np
import torch

# FIXED CONFIG VALUES:
# splitting the excerpt in 5 equal segments:
TEMPORAL_SEGMENTS = {'type': 'manual', 'manual_segments': [(0, 7246), (7246, 14492), (14492, 21738), (21738, 28984),
                                                           (28984, 36230)]}

# fixing the number of samples after the stability evaluation (not included in this repo):
NUM_SAMPLES = 8192

# fixing the threshold for the provided model
THRESHOLD = 0.51

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def setup(model_file, magscale, firstconv_zeromean):
    """This setup method prepares the model, the spectrogram transform and the config."""
    # first set up config dict
    cfg = {}
    cfg.update(parse_config_file(default_vars_path))
    cfg['magscale'] = magscale
    cfg['arch.firstconv_zeromean'] = firstconv_zeromean

    hop_size = cfg['sample_rate'] // cfg['fps']
    spec_transform = Spectrogram(n_fft=cfg['frame_len'], hop_length=hop_size, power=1.)

    bin_nyquist = cfg['frame_len'] // 2 + 1
    bin_mel_max = bin_nyquist * 2 * cfg['mel_max'] // cfg['sample_rate']
    spec_len = bin_mel_max

    net_config = prep_net_config(cfg, spec_len)
    model = load_model(model_file, SingingVoiceDetector(net_config), device)
    return model, spec_transform, cfg


def compute_explanation(spec, composition_fn_specLIME, predict_fn, sample_rate, hop_size,
                        baseline="min", targets=None, sort_by_index=True):
    """This function computes an explanation for a given spectrogram."""
    if targets is None:
        targets = [1]
    explanations = []
    audio_explainer = LimeAudioExplainer(absolute_feature_sort=False, verbose=False)

    factorization = TimeFrequencyTorchFactorization(spec,
                                                    frequency_segments=4,
                                                    temporal_segmentation_params=TEMPORAL_SEGMENTS,
                                                    hop_length=hop_size,
                                                    target_sr=sample_rate,
                                                    composition_fn=composition_fn_specLIME, baseline=baseline)
    explanation = audio_explainer.explain_instance(factorization, predict_fn, labels=[0, 1],
                                                   num_samples=NUM_SAMPLES, batch_size=1)

    for tix in targets:
        _, used_indeces = explanation.get_sorted_components(tix, positive_components=True, negative_components=False,
                                                            num_components=3, return_indeces=True)
        explanations.append(factorization.retrieve_components(used_indeces))

    metrics = {'fidelity': explanation.score[1],
               'local': explanation.local_pred[1],
               'global': explanation.neighborhood_labels[0, 1],
               'intercept': explanation.intercept[1]}

    weights = [w for w in explanation.local_exp[1]]
    if sort_by_index:
        weights.sort(key=lambda w: w[0])
    return explanations, metrics, weights


def compute_all_modifications(original, adversarial, compose_and_predict, result_tuple, binary_factorization,
                              k_components, target_idx, compute_norm_on="adversary", verbose=False):
    """This function computes modified spectrograms (by removing components determined by LIME or
    based on the norm) and the predictions for those modified spectrograms."""
    assert target_idx in [0, 1]
    assert compute_norm_on in ["adversary", "delta"]

    original_pred = compose_and_predict(original).flatten()
    adversarial_pred = compose_and_predict(adversarial).flatten()

    if verbose:
        print(original_pred, adversarial_pred)

    delta = adversarial - original

    (_, _, _), (_, _, weights_adv) = result_tuple
    # for the negative class we take the inverted weights
    weights_adv.sort(key = lambda w: w[1], reverse=True * target_idx)
    if k_components == 'positive':
        if verbose: print('taking all positive components')
        temp_weights = np.array([w[1] for w in weights_adv])
        if target_idx == 1:
            k_components = (np.array(temp_weights) > 0).sum()
        else:
            k_components = (np.array(temp_weights) < 0).sum()
        if verbose: print('{} positive components'.format(k_components))

    if verbose:
        print("sorted weights:", weights_adv[:k_components])

    top_weights = [w[0] for w in weights_adv][:k_components]

    mod_delta_slime = delta * binary_factorization.retrieve_components(top_weights)  # -> 2, 0
    pred_slime = compose_and_predict(original + mod_delta_slime).flatten()

    norm_input = adversarial
    if compute_norm_on == "delta":
        norm_input = delta

    # not nice but we never changed hop_size or sample_rate ...
    binary_mask, largest_indeces = delta_to_binary_mask(norm_input, k_components, hop_size=315, sample_rate=22050,
                                                        return_indeces=True)

    mod_delta_obv = delta * binary_mask
    pred_obv = compose_and_predict(original + mod_delta_obv).flatten()

    if verbose:
        print(pred_slime, pred_obv)

    return original_pred, adversarial_pred, pred_slime, pred_obv, mod_delta_slime, mod_delta_obv, top_weights, largest_indeces


def delta_to_binary_mask(delta, k, hop_size, sample_rate, return_indeces=False):
    """returns a binary mask indicating the k segments with the largest norm / magnitude """
    delta_factorization = TimeFrequencyTorchFactorization(delta,
                                                          frequency_segments=4,
                                                          temporal_segmentation_params=TEMPORAL_SEGMENTS,
                                                          hop_length=hop_size,
                                                          target_sr=sample_rate,
                                                          baseline="zero")

    # for this we use the TimeFrequencyTorchFactorization and fill it with ones and baseline 'zero',
    # because then retrieving segments will result in a binary mask
    # and we can ensure that it uses exactly the same "grid" for segmentation as we use for the LIME
    # algorithm
    binary_factorization = TimeFrequencyTorchFactorization(torch.ones_like(delta),
                                                           frequency_segments=4,
                                                           temporal_segmentation_params=TEMPORAL_SEGMENTS,
                                                           hop_length=hop_size,
                                                           target_sr=sample_rate,
                                                           baseline="zero")
    segment_values = []
    for i in range(binary_factorization.get_number_components()):
        single_components = delta_factorization.retrieve_components([i])
        segment_values.append((single_components ** 2).sum())
    ranked = np.argsort(segment_values)
    largest_indices = ranked[::-1][:k]
    binary_mask = binary_factorization.retrieve_components(largest_indices)
    if return_indeces:
        return binary_mask, largest_indices
    return binary_mask


def evaluate_component_impact(adversaries, explanations, compose_and_predict, binary_factorization,
                              threshold, k_components=3, target_idx=-1, compute_norm_on="adversary", verbose=False):
    """This function computes which examples can be flipped by using either LIME or the norm
    for component selection."""

    flipped_slime = []
    flipped_norm_based = []
    for i in range(len(adversaries)):
        for j in range(len(adversaries[i])):
            original_spec, adversarial_spec = adversaries[i][j]
            explanation = explanations[i][j]

            original_pred, adversarial_pred, pred_slime, pred_obv, mod_delta_slime, mod_delta_obv, top_weights, largest_indeces = \
                compute_all_modifications(original_spec, adversarial_spec, compose_and_predict, explanation,
                                          binary_factorization, k_components=k_components, target_idx=target_idx,
                                          compute_norm_on=compute_norm_on, verbose=verbose)

            if target_idx == 1:
                assert original_pred[1] < threshold  # sanity check
                assert adversarial_pred[1] > threshold  # sanity check

                flipped_slime.append(pred_slime[1] > threshold)
                flipped_norm_based.append(pred_obv[1] > threshold)
            else:
                assert original_pred[1] > threshold  # sanity check
                assert adversarial_pred[1] < threshold  # sanity check

                flipped_slime.append(pred_slime[1] < threshold)
                flipped_norm_based.append(pred_obv[1] < threshold)

    flipped_slime = np.array(flipped_slime)
    flipped_norm_based = np.array(flipped_norm_based)

    return flipped_slime, flipped_norm_based


def create_composition_predict_fn_explanation(model, take_first_element=False):
    """This function prepares the prediction function which is used by LIME
    and the composition function which is used by the factorization object to prepare th emodel input."""
    def predict_fn_specLIME(x):
        if take_first_element:
            x = x[0]
        predictions = model.probabilities_from_preprocessed_spec(x)
        x_pred = predictions.detach().cpu().numpy().item()
        return np.array([[1 - x_pred, x_pred]])

    def composition_fn_specLIME(x):
        x = x.transpose(0, 1).unsqueeze(0)
        x = model.pre_proc(x)
        x = model.mag_trans(x)
        return x

    return composition_fn_specLIME, predict_fn_specLIME


def preprocess_adversaries(adversaries, prep_fn, **kwargs):
    """This function applies a preprocessing function to a list of adversaries."""
    processed_adversaries = []
    for adv_list in adversaries:
        temp_list = []
        for adv in adv_list:
            temp_list.append((prep_fn(adv[0], **kwargs), prep_fn(adv[1], **kwargs)))
        processed_adversaries.append(temp_list)
    return processed_adversaries


def prepare_factorization_input_spec(x, model, to_tensor=False):
    if to_tensor:
        x = torch.tensor(x).to(device)
    x = x.unsqueeze(0)
    x = model.pre_proc(x)
    x = model.mag_trans(x)
    x = x.squeeze().transpose(0, 1)
    return x


def npyfy(x):
    return x.detach().cpu().numpy()


def composition_fn_specLIME(x):
    return x.transpose(0, 1).unsqueeze(0)
