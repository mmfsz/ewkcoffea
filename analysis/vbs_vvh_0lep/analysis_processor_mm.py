#!/usr/bin/env python
#import sys
import coffea
import numpy as np
import awkward as ak
import copy
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
import hist
from hist import axis
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.corrections as cor_tc

from ewkcoffea.modules.paths import ewkcoffea_path as ewkcoffea_path
import ewkcoffea.modules.selection_wwz as es_ec
import ewkcoffea.modules.objects_wwz as os_ec
import ewkcoffea.modules.corrections as cor_ec

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_ec_param = GetParam(ewkcoffea_path("params/params.json"))



class AnalysisProcessor(processor.ProcessorABC):

    #def __init__(self, samples={}, hist_lst=None):
    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, skip_obj_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False):
        self._samples = samples

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "met"   : axis.Regular(180, 0, 2000, name="met", label="MET"),
            "higgs_pt"   : axis.Regular(180, 0, 4000, name="higgs_pt", label="Higgs $p_T$"),
        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        self._processes = ["sig","data","QCD"]
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                #hist.axis.StrCategory([], growth=True, name="systematic", label="systematic"),
                hist.axis.StrCategory([], growth=True, name="lherwgt_idx", label="lherwgt_idx"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight", # Keeps track of sumw2
                name="Counts",
            )

        # Set the accumulator
        self._accumulator = processor.dict_accumulator(dout)

        # Set the list of hists to fill
        if hist_lst is None:
            # If the hist list is none, assume we want to fill all hists
            self._hist_lst = list(self._accumulator.keys())
        else:
            # Otherwise, just fill the specified subset of hists
            for hist_to_include in hist_lst:
                if hist_to_include not in self._accumulator.keys():
                    raise Exception(f"Error: Cannot specify hist \"{hist_to_include}\", it is not defined in the processor.")
            self._hist_lst = hist_lst # Which hists to fill


    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def postprocess(self, accumulator):
        return accumulator


    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters

        isSignal = False
        if ak.any(events["LHEReweightingWeight"]):
           isSignal = True

        ######### Objects and cuts definitions ###########
        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.Array([1] * len(events))
        
        print(events.fields)
        # Retrieve or build new variables from input tree
        met  = events.MET
        #met  = events.MET_pt
        higgs = events.Higgs
        print("higgs.pt", higgs.pt)
        print("higgs.mass", higgs.mass)



        ######### Store boolean masks with PackedSelection ##########
        selections = PackedSelection(dtype='uint64')

        selections.add("all_events", ak.Array([True] * len(events)))
        cat_dict = {
          "Run2AllHad3FJ" : [
              "all_events",
              #"Pass_Cut9",
          ],
          "Run2AllHadSemiMerg" : [
              "all_events",
          ],
          "SPANetStudy" : [
              "all_events",
          ],
        }        

        ######### Fill histos #########    
        # Put the variables we'll plot into a dictionary for easy access
        dense_variables_dict = {
                #"met" : met.pt,
                #"metphi" : met.phi,
                "higgs_pt": higgs.pt,
                "AK4V1DeltaR": events.AK4V1DeltaR,
        }

        # Loop over histograms
        for dense_axis_name, dense_axis_vals in dense_variables_dict.items():
            if dense_axis_name not in self._hist_lst:
                print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                continue

            # Loop over processes and categories
            for process in ["sig"]: #self._processes:
                mask_process = ak.Array([process == "sig"] * len(events))  # Broadcast to array
                for sr_cat, cuts_lst in cat_dict.items():
                    # Initialize weight
                    weight = events.nom if dense_axis_name.endswith("_counts") else events.weight

                    all_cuts_mask = selections.all(*cuts_lst)
                    final_mask = all_cuts_mask & mask_process
                    print("Type of final_mask:", ak.type(final_mask))
                    print("Length of final_mask:", len(final_mask))
                    print("Number of events passing final_mask:", ak.sum(final_mask))

                    axes_fill_info_dict = {
                            dense_axis_name: ak.fill_none(dense_axis_vals[all_cuts_mask],0),
                            "weight": ak.fill_none(weight[all_cuts_mask],0),    
                            "process": process,
                            "category": sr_cat,
                            "lherwgt_idx": "nominal",
                        }
                    self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

                    # if isSignal:
                    #     n_rwgt_wgt = len(events["LHEReweightingWeight"][0])
                    #     #print("Number of reweighting weights:", n_rwgt_wgt)
                    #     #print("Type of events['LHEReweightingWeight']:", ak.type(events["LHEReweightingWeight"]))
                    #     for i in range(n_rwgt_wgt):
                    #         rwgt_wgt = events["LHEReweightingWeight"][:, i]

                    #         # Apply mask before multiplication
                    #         if ak.any(final_mask):
                    #             final_weight = (weight * rwgt_wgt)[final_mask]
                    #             final_vals = ak.fill_none(dense_axis_vals[final_mask], 0)
                    #         else:
                    #             print("Warning: final_mask selects no events, using empty arrays")
                    #             final_weight = ak.Array([])
                    #             final_vals = ak.Array([])

                    #         # Fill histogram
                    #         axes_fill_info_dict = {
                    #             dense_axis_name: final_vals,
                    #             "weight": final_weight,
                    #             "process": process,
                    #             "category": sr_cat,
                    #             "lherwgt_idx": f"idx{i}",
                    #         }
                    #         self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)
                    # else:
                    #     # Non-signal case
                    #     if ak.any(final_mask):
                    #         final_weight = weight[final_mask]
                    #         final_vals = ak.fill_none(dense_axis_vals[final_mask], 0)
                    #     else:
                    #         print("Warning: final_mask selects no events, using empty arrays")
                    #         final_weight = ak.Array([])
                    #         final_vals = ak.Array([])
                    #     # Fill histogram
                    #     axes_fill_info_dict = {
                    #         dense_axis_name: final_vals,
                    #         "weight": final_weight,
                    #         "process": process,
                    #         "category": sr_cat,
                    #         "lherwgt_idx": "nominal",
                    #     }
                    #     self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

        return self.accumulator