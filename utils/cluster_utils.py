from copy import deepcopy

from ray.tune import TuneError
from ray.tune.config_parser import create_trial_from_spec
from ray.tune.suggest import BasicVariantGenerator
from ray.tune.suggest.variant_generator import generate_variants, flatten_resolved_vars, format_vars


class CustomVariantGenerator(BasicVariantGenerator):
    def __init__(self, shuffle=False):
        super(CustomVariantGenerator, self).__init__(shuffle=shuffle)

    @staticmethod
    def _extract_resolved_base_vars(cfg, cfgs):
        resolved = {}
        for k, v in cfg.items():
            if k == "grid_search": continue
            if isinstance(v, dict):
                for part_k, part_v in CustomVariantGenerator._extract_resolved_base_vars(cfg[k],
                                                                                         [c[k] for c in cfgs]).items():
                    if isinstance(part_k, tuple):
                        resolved.update({(k, *part_k): part_v})
                    else:
                        resolved.update({(k, part_k): part_v})
            else:
                is_same = True
                for c in cfgs:
                    if not isinstance(c, dict) or k not in c:
                        is_same = False
                if is_same:
                    for v2 in [c[k] for c in cfgs]:
                        if v != v2: is_same = False
                if not is_same:
                    resolved.update({(k): v})

        return resolved

    def _generate_trials(self, num_samples, unresolved_spec, output_path=""):
        """Generates Trial objects with the variant generation process.

        Uses a fixed point iteration to resolve variants. All trials
        should be able to be generated at once.

        See also: `ray.tune.suggest.variant_generator`.

        Yields:
            Trial object
        """

        if "run" not in unresolved_spec:
            raise TuneError("Must specify `run` in {}".format(unresolved_spec))
        for _ in range(num_samples):
            # Iterate over list of configs
            for unresolved_cfg in unresolved_spec["config"]:
                unresolved_spec_variant = deepcopy(unresolved_spec)
                unresolved_spec_variant["config"] = unresolved_cfg
                resolved_base_vars = CustomVariantGenerator._extract_resolved_base_vars(unresolved_cfg,
                                                                                        unresolved_spec["config"])
                print("Resolved base cfg vars", resolved_base_vars)
                for resolved_vars, spec in generate_variants(unresolved_spec_variant):
                    resolved_vars.update(resolved_base_vars)
                    print("Resolved vars", resolved_vars)
                    trial_id = "%05d" % self._counter
                    experiment_tag = str(self._counter)
                    if resolved_vars:
                        experiment_tag += "_{}".format(
                            format_vars({k: v for k, v in resolved_vars.items() if "tag" in k}))
                    self._counter += 1
                    yield create_trial_from_spec(
                        spec,
                        output_path,
                        self._parser,
                        evaluated_params=flatten_resolved_vars(resolved_vars),
                        trial_id=trial_id,
                        experiment_tag=experiment_tag)

    def _generate_resolved_specs(self, num_samples, unresolved_spec):
        """Needed for slurm_cluster.py
        """
        for _ in range(num_samples):
            # Iterate over list of configs
            for unresolved_cfg in unresolved_spec["config"]:
                unresolved_spec_variant = deepcopy(unresolved_spec)
                unresolved_spec_variant["config"] = unresolved_cfg
                resolved_base_vars = CustomVariantGenerator._extract_resolved_base_vars(unresolved_cfg,
                                                                                        unresolved_spec["config"])
                print("Resolved base cfg vars", resolved_base_vars)
                for resolved_vars, spec in generate_variants(unresolved_spec_variant):
                    resolved_vars.update(resolved_base_vars)
                    print("Resolved vars", resolved_vars)
                    trial_id = "%05d" % self._counter
                    experiment_tag = str(self._counter)
                    if resolved_vars:
                        experiment_tag += "_{}".format(
                            format_vars({k: v for k, v in resolved_vars.items() if "tag" in k}))
                    self._counter += 1
                    yield {
                        "spec": spec,
                        "evaluated_params": flatten_resolved_vars(resolved_vars),
                        "trial_id": trial_id,
                        "experiment_tag": experiment_tag
                    }