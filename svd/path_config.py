import os

project_root = '/path/to/your/project_dir/' # this is where the results will be stored
experiments_path = os.path.join(project_root, 'experiments')
dataset_path = "/path/to/jamendo/audio/"
cache_path = os.path.join(project_root, "data_cache/")
# set the path to the location of your 'horses' directory
horses_path = 'horses'  # you can leave it like this if you execute the code from the 'veracity' directory
# set the path to default.vars
default_vars_path = '/path/to/default.vars'

# create paths if they don't exist:
if not os.path.exists(experiments_path):
    os.mkdir(experiments_path)

if not os.path.exists(cache_path):
    os.mkdir(cache_path)

if not os.path.exists(os.path.join(project_root, "adversaries")):
    os.mkdir(os.path.join(project_root, "adversaries"))

if not os.path.exists(os.path.join(project_root, "figures")):
    os.mkdir(os.path.join(project_root, "figures"))
    os.mkdir(os.path.join(project_root, "figures", "experiment_fidelity"))
    os.mkdir(os.path.join(project_root, "figures", "experiment_flip_count"))
    os.mkdir(os.path.join(project_root, "figures", "experiment_subset_perturbation"))
    os.mkdir(os.path.join(project_root, "figures", "experiment_horses"))


def build_adversary_path(model_name, attack_type, target, subset):
    return os.path.join(experiments_path, model_name, "adversaries_{}_{}_{}.pt".format(attack_type, target, subset))


def build_stability_path(model_name, attack_type, target, subset):
    return os.path.join(experiments_path, model_name, "stability_{}_{}_{}.pt".format(attack_type, target, subset))


def build_adversarial_examples_path(attack_type, first_layer, subset):

    type_param_mapping = {
        'spec': 'cw_lr5e4_alpha15_eps1e1_fa1_nopool',
        'wave': 'cw_lr3e4_alpha2_eps1e2_fa1_nopool'
    }

    return os.path.join(project_root, 'adversaries', '{}_{}_{}_{}'.format(
        attack_type,
        type_param_mapping[attack_type],
        first_layer.replace("0", "zero"),
        subset
        ))


def build_explanation_path(model_name, baseline, attack_type, target, subset):
    return os.path.join(experiments_path, model_name, "explanations_{}_kw025_{}_{}_{}.pt".format(baseline, attack_type, target, subset))


def build_model_path(model_name):
    return os.path.join(experiments_path, model_name, 'model.pt')


def build_localized_explanations_path(model_name, baseline, attack_type, target, subset, k):
    return os.path.join(experiments_path, model_name,
                        "components_localized_{}_{}_{}_{}_k={}.pt".format(baseline, attack_type, target, subset, k))


def build_figure_perturbation_path(baseline, attack_type, target, subset, cumulative):
    return os.path.join(project_root, 'figures', 'experiment_subset_perturbation',
                        "perturbation_{}_{}_{}_{}_{}.png".format(baseline, attack_type, target, subset, cumulative))


def build_figure_weight_path(baseline, attack_type, target, subset):
    return os.path.join(project_root, 'figures', 'experiment_weight',
                        "weight_{}_{}_{}_{}.png".format(baseline, attack_type, target, subset))


def build_figure_flipcount_path(baseline, attack_type, target, subset):
    return os.path.join(project_root, 'figures', 'experiment_flip_count',
                        "flipcount_{}_{}_{}_{}.png".format(baseline, attack_type, target, subset))
