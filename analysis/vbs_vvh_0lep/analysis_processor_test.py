import yaml
import awkward as ak
import numpy as np
import coffea.processor as processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection
import hist
from configs.selections_config import get_cutflow
ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

from config_dense_vars import dense_variables_config, objects_config, derived_objects_config

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, project=None, cutflow_names=None):

        # Load full cutflow config
        # a dictionary: cutflow_dict[cf_name] = (mode, cuts_dict)
        if project is None: 
            project = "default"
        full_cutflows = get_cutflow(f'configs/cutflow_{project}.yaml', None)
        if cutflow_names is None:
            # Use all
            self.cutflows = full_cutflows
        else:
            # Filter to requested list
            self.cutflows = {name: full_cutflows[name] for name in cutflow_names if name in full_cutflows}

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            var_name: cfg["axis"]
            for var_name, cfg in dense_variables_config.items()
        }

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            var_name: cfg["axis"]
            for var_name, cfg in dense_variables_config.items()
        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                hist.axis.StrCategory([], growth=True, name="systematic", label="systematic"),
                hist.axis.StrCategory([], growth=True, name="year", label="year"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight", # Keeps track of sumw2
                name="Counts",
            )

        self._accumulator = processor.dict_accumulator(dout)
        
    @property
    def accumulator(self):
        return self._accumulator        

    def process(self, events):
        
        ######### Metadata ###########
        
        json_name = events.metadata["dataset"]
        print(f"json_name = {json_name}")
        isSig = "sig" in json_name
        isData = "data" in json_name


        ######### Objects and variables ###########

        # Build independent objects from config
        objects = {
            name: builder(events)
            for name, builder in objects_config.items()
        }
        # Build derived objects from config
        objects.update({
            name: builder(events,objects)
            for name, builder in derived_objects_config.items()
        })

        print(events.fields)
        # print(objects["leadAK8"])
        # print(objects["leadAK8"].fields)
        # lead_ak8_pt = objects["leadAK8"].pt
        # print(f"leadAK8.pt = {lead_ak8_pt}")

        variables  = {
            var_name: cfg["expr"](events, objects)
            for var_name, cfg in dense_variables_config.items()
        }
        dense_variables_dict  = variables

        ######### Normalization and weights ###########

        # Weights 
        n_events = len(events)
        ones = ak.Array([np.float32(1.0)] * n_events)
        zeros = ak.Array([np.float32(0.0)] * n_events)
        weights = Weights(n_events)
        weights.add("genweight", events.genWeight if "genWeight" in events.fields else ones)

        wgt_var_dict = {
            "nominal": events.weight,
            "count"  : ones,
        }

        # if isSig:                 # or use the ak.firsts trick
        #     print("Add other coupling points")
        #     #wgt_var_dict["SM"] = events.weight * ak.fill_none(events.LHEReweightingWeight[:,60],0)
        #     #wgt_var_dict["c2v_1p5"] = events.weight * ak.fill_none(events.LHEReweightingWeight[:,72],0)
        #     daughters = VV_ndaughters(events)
        #     n_had = ak.fill_none(daughters['n_had'], 0)
        #     n_MET = ak.fill_none(daughters['n_MET'], 0)
        #     mask1 = (n_had == 2) & (n_MET == 2)
        #     mask2 = (n_had == 2) & (n_MET == 1)
        #     wgt_var_dict["c2v_1p5_qqNuNu"] = events.weight * mask1
        #     wgt_var_dict["c2v_1p5_qqlNu"] = events.weight * mask2

        
        ######### Selections ##########

       
        
        # # Define categories (as in your code: all_events, exactly0lep, etc.)
        # selection.add("all_events", ak.ones_like(events.event, dtype=bool))
        
        # # Leptons (simplified; assume combined Electron+Muon, with masks)
        # ele = events.Electron
        # mu = events.Muon
        # lep = ak.concatenate([ele, mu], axis=1)
        # lep = lep[ak.argsort(lep.pt, ascending=False)]  # Sort by pt
        # nleps = ak.num(lep)
        
        # selection.add("exactly0lep", nleps == 0)
        # selection.add("exactly1lep", nleps == 1)
        # selection.add("geq2lep", nleps >= 2)
        
        # # Jets and Fatjets (simplified; add your full definitions)
        # jets = events.Jet
        # njets = ak.num(jets)
        # jets["is_central"] = abs(jets.eta) < 2.5  # Example
        # jets_central = jets[jets.is_central]
        # jets_forward = jets[~jets.is_central]
        
        # fatjets = events.FatJet  # Assume AK8
        # nfatjets = ak.num(fatjets)
        
        # selection.add("exactly0fj", nfatjets == 0)
        # selection.add("exactly1fj", nfatjets == 1)
        # selection.add("exactly2fj", nfatjets == 2)
        # selection.add("geq2fj", nfatjets >= 2)
        # selection.add("leq2fj", nfatjets <= 2)
        
        # # Combined categories (as in your code)
        # selection.add("exactly0lep_exactly1fj", selection.all("exactly0lep", "exactly1fj"))
        # selection.add("exactly0lep_exactly2fj", selection.all("exactly0lep", "exactly2fj"))
        # selection.add("exactly0lep_geq2fj", selection.all("exactly0lep", "geq2fj"))
        # selection.add("exactly0lep_leq2fj", selection.all("exactly0lep", "leq2fj"))
        
        # # More derivations (add as needed from your code)
        # met = ones #events.MET FIXME
        # scalarptsum_lep = ak.sum(lep.pt, axis=1)
        # # etc. for other scalars
        

        # # Similar for others
        
        # fatjets_padded =  ak.pad_none(fatjets, 2)
        # fj0 = fatjets_padded[:, 0]
        # fj1 = fatjets_padded[:, 1]

        ######### Selections and fill histograms ##########
        # Note: conceptually these are two separate steps. However, to avoid storing all the masks and the fill the histograms, it is more efficient to fill the histogram as we go and then discard the masks. 
        
        for cf_name, cf_data in self.cutflows.items():
            mode, cutflow_steps = cf_data  # Unpack mode and steps
            
            selections = PackedSelection(dtype='uint64')
            
            for sel, crit_fn in cutflow_steps.items():
                mask = crit_fn(events, variables, objects)
                selections.add(sel, mask)
            
            # Define base selections
            # If defined in the cutflow, these selections are always applied irrespectively of the mode, to ensure distributions make sense
            base_sels = []
            if 'objsel' in cutflow_steps:
                base_sels.append('objsel')
            elif 'all_events' in cutflow_steps:
                base_sels.append('all_events')
            
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
            

            # Produce histograms for each weight variation and each selection
            for wgt_key, wgt in wgt_var_dict.items():
                for sel, sel_mask in cut_mask_dict.items():
                    for dense_axis_name, dense_axis_vals in dense_variables_dict.items():
                        axes_fill_info_dict = {
                            dense_axis_name: ak.fill_none(dense_axis_vals[sel_mask], 0),
                            "weight": ak.fill_none(wgt[sel_mask], 0),
                            "process": ak.fill_none(events.namewithyear[sel_mask], "unknown"),
                            "category": f"{cf_name}_{sel}",  # Prefix with cutflow name
                            "systematic": wgt_key,
                        }
                        if "year" in self.accumulator[dense_axis_name].axes.name:
                            axes_fill_info_dict["year"] = ak.fill_none(events.year[sel_mask], "unknown")
                        
                        self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

        
        
        return self.accumulator

    def postprocess(self, accumulator):
        return accumulator