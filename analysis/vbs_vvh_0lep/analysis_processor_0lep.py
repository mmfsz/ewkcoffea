import yaml
import awkward as ak
import numpy as np
import coffea.processor as processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection
import hist
import time
import sys
from configs.selections_config import get_cutflow
ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

from configs.config_dense_vars import (
    objects_config, derived_objects_config, derived_objects_config_level2,
    derived_objects_config_level3, derived_objects_config_level4,
    dense_variables_config, dense_truth_variables_config, truth_objects_config
)

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, project=None, cutflow_names=None, hist_names=None, debug=False, store_truth=False):

        self._store_truth = store_truth
        self._debug = debug
        if self._debug:
            self._timing_log = open("timing.log", "w")

        # Load full cutflow config
        # a dictionary: cutflow_dict[cf_name] = (mode, cuts_dict)
        if project is None:
            project = "default"
        full_cutflows = get_cutflow(f'configs/cutflow_{project}.yaml', None)
        if cutflow_names is None:
            # Use all
            self.cutflows = full_cutflows
        else:
            # Handle both string and list inputs
            if isinstance(cutflow_names, str):
                cutflow_names = [cutflow_names]
            # Filter to requested list
            self.cutflows = {name: full_cutflows[name] for name in cutflow_names if name in full_cutflows}

        # Determine which histograms to fill
        if hist_names is None:
            # Use all histograms
            self._hist_names = list(dense_variables_config.keys())
        else:
            # Handle both string and list inputs
            if isinstance(hist_names, str):
                hist_names = [hist_names]
            # Filter to only requested histograms that exist
            self._hist_names = [name for name in hist_names if name in dense_variables_config]

        # Conditionally add truth hists
        if self._store_truth:
            self._hist_names.extend(list(dense_truth_variables_config.keys()))

        # Create the dense axes for the histograms (only for selected histograms)
        # Combine both config dicts to look up axes
        all_variables_config = {**dense_variables_config, **dense_truth_variables_config}
        self._dense_axes_dict = {
            var_name: all_variables_config[var_name]["axis"]
            for var_name in self._hist_names
        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                hist.axis.StrCategory([], growth=True, name="systematic", label="systematic"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight",
                name="Counts",
            )

        self._accumulator = processor.dict_accumulator(dout)

    @property
    def accumulator(self):
        return self._accumulator

    def _log(self, msg):
        """Log timing message if debug mode is enabled"""
        if self._debug:
            print(msg, file=self._timing_log, flush=True)

    def process(self, events):

        ######### Metadata ###########

        json_name = events.metadata["dataset"]
        isSig = "sig" in json_name
        isData = "data" in json_name
        
        # Truth handling block
        # NanoAODSchema groups branches by prefix, so truth_h_idx becomes events.truth.h_idx
        has_truth = 'truth' in events.fields and hasattr(events.truth, 'h_idx')
        compute_truth = self._store_truth and has_truth

        ######### Objects and variables ###########

        t0 = time.time()
        # Build independent objects from config
        objects = {
            name: builder(events)
            for name, builder in objects_config.items()
        }
        # Build derived objects from config (level 1)
        objects.update({
            name: builder(events, objects)
            for name, builder in derived_objects_config.items()
        })
        # Build derived objects from config (level 2 - depend on level 1)
        objects.update({
            name: builder(events, objects)
            for name, builder in derived_objects_config_level2.items()
        })
        # Build derived objects from config (level 3 - depend on level 2)
        objects.update({
            name: builder(events, objects)
            for name, builder in derived_objects_config_level3.items()
        })
        # Build derived objects from config (level 4 - pre-computed quantities)
        objects.update({
            name: builder(events, objects)
            for name, builder in derived_objects_config_level4.items()
        })
        # Build truth-matched objects (signal only)
        if compute_truth:
            objects.update({
                name: builder(events, objects)
                for name, builder in truth_objects_config.items()
            })
        self._log(f"[TIMING] Objects built in {time.time()-t0:.2f}s")

        t0 = time.time()
        # Only compute variables for histograms we're filling
        dense_variables_dict = {
            var_name: dense_variables_config[var_name]["expr"](events, objects)
            for var_name in self._hist_names
            if var_name in dense_variables_config
        }
        # Add truth variables if computing truth
        if compute_truth:
            dense_variables_dict.update({
                var_name: dense_truth_variables_config[var_name]["expr"](events, objects)
                for var_name in self._hist_names
                if var_name in dense_truth_variables_config
            })
        self._log(f"[TIMING] Variables built in {time.time()-t0:.2f}s")


        ######### Normalization and weights ###########

        # Weights
        n_events = len(events)
        ones = ak.Array([np.float32(1.0)] * n_events)
        weights = Weights(n_events)
        weights.add("genweight", events.genWeight if "genWeight" in events.fields else ones)

        wgt_var_dict = {
            "nominal": events.weight,
            #"count"  : ones,
        }

        ######### Selections and fill histograms ##########
        self._log(f"[TIMING] Starting cutflow loop with {len(self.cutflows)} cutflows")

        for cf_name, cf_data in self.cutflows.items():
            self._log(f"[TIMING] Processing cutflow: {cf_name}")
            mode, cutflow_steps = cf_data  # Unpack mode and steps

            selections = PackedSelection(dtype='uint64')

            for sel, crit_fn in cutflow_steps.items():
                mask = crit_fn(events, dense_variables_dict, objects)
                selections.add(sel, mask)

            # Define base selections
            base_sels = []
            if 'all_events' in cutflow_steps:
                base_sels.append('all_events')
            if 'objsel' in cutflow_steps:
                base_sels.append('objsel')

            # Build cut mask dictionary based on mode
            cut_mask_dict = {}
            if mode == 'cumulative':
                cumulative_cuts = list(base_sels)
                for sel in cutflow_steps:
                    if sel not in base_sels:
                        cumulative_cuts.append(sel)
                    cut_mask_dict[sel] = selections.all(*cumulative_cuts)
            elif mode == 'n_minus_1':
                for sel in cutflow_steps:
                    exclusive_sels = [k for k in cutflow_steps if k != sel] + base_sels
                    cut_mask_dict[sel] = selections.all(*exclusive_sels)
            elif mode == 'individual':
                for sel in cutflow_steps:
                    individual_sels = base_sels + [sel] if sel not in base_sels else base_sels
                    cut_mask_dict[sel] = selections.all(*individual_sels)
            else:
                raise ValueError(f"Unknown mode for {cf_name}: {mode}")


            # Produce histograms for each selection and weight variation
            t_fill_start = time.time()
            for sel, sel_mask in cut_mask_dict.items():
                t_mask = time.time()
                # Pre-compute masked values for this selection
                masked_vars = {
                    var_name: np.asarray(ak.fill_none(var_vals[sel_mask], 0))
                    for var_name, var_vals in dense_variables_dict.items()
                }
                masked_process = np.asarray(ak.fill_none(events.namewithyear[sel_mask], "unknown"))
                self._log(f"[TIMING] Masking for {cf_name}_{sel} took {time.time()-t_mask:.2f}s")

                t_inner = time.time()
                category_str = f"{cf_name}_{sel}"
                for wgt_key, wgt in wgt_var_dict.items():
                    masked_weight = np.asarray(ak.fill_none(wgt[sel_mask], 0))

                    for dense_axis_name, masked_val in masked_vars.items():
                        self.accumulator[dense_axis_name].fill(
                            **{dense_axis_name: masked_val},
                            weight=masked_weight,
                            process=masked_process,
                            category=category_str,
                            systematic=wgt_key,
                        )
                self._log(f"[TIMING] Filling for {cf_name}_{sel} took {time.time()-t_inner:.2f}s")
            self._log(f"[TIMING] Fill loop for {cf_name} took {time.time()-t_fill_start:.2f}s")


        return self.accumulator

    def postprocess(self, accumulator):
        return accumulator
