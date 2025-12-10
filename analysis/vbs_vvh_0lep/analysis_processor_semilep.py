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

from topcoffea.modules.paths import topcoffea_path
#import topcoffea.modules.event_selection as es_tc
#import topcoffea.modules.corrections as cor_tc

from ewkcoffea.modules.paths import ewkcoffea_path as ewkcoffea_path
#import ewkcoffea.modules.selection_wwz as es_ec
import ewkcoffea.modules.objects_wwz as os_ec
import ewkcoffea.modules.corrections as cor_ec

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_ec_param = GetParam(ewkcoffea_path("params/params.json"))


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_systematics=False, skip_obj_systematics=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "met"   : axis.Regular(180, 0, 750, name="met",  label="met"),
            "metphi": axis.Regular(180, -3.1416, 3.1416, name="metphi", label="met phi"),
            "scalarptsum_jetCentFwd" : axis.Regular(180, 0, 2000, name="scalarptsum_jetCentFwd", label="H_T small radius"),
            "scalarptsum_jetFwd" : axis.Regular(180, 0, 1000, name="scalarptsum_jetFwd", label="H_T forward"),
            "scalarptsum_jetCent" : axis.Regular(180, 0, 2000, name="scalarptsum_jetCent", label="H_T central"),
            "scalarptsum_lep" : axis.Regular(180, 0, 2000, name="scalarptsum_lep", label="S_T"),
            "scalarptsum_lepmet" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmet", label="S_T + metpt"),
            "scalarptsum_lepmetFJ" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ", label="S_T + metpt + FJ pt"),
            "scalarptsum_lepmetFJ10" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ10", label="S_T + metpt + FJ0 + FJ1 pt"),
            "scalarptsum_lepmetalljets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetalljets", label="S_T + metpt + H_T all"),
            "scalarptsum_lepmetcentjets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetcentjets", label="S_T + metpt + H_T cent"),
            "scalarptsum_lepmetfwdjets" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmetfwdjets", label="S_T + metpt + H_T fwd"),
            "l0_pt"  : axis.Regular(180, 0, 1000, name="l0_pt", label="l0 pt"),
            "l0_eta"  : axis.Regular(180, -3,3, name="l0_eta", label="l0 eta"),
            "l1_pt"  : axis.Regular(180, 0, 1000, name="l1_pt", label="l1 pt"),
            "l1_eta"  : axis.Regular(180, -3,3, name="l1_eta", label="l1 eta"),
            "l2_pt"  : axis.Regular(180, 0, 1000, name="l2_pt", label="l2 pt"),
            "l2_eta"  : axis.Regular(180, -3,3, name="l2_eta", label="l2 eta"),

            "mass_l0l1"  : axis.Regular(180, 0,1000, name="mass_l0l1", label="mll of leading two leptons"),
            "dr_l0l1" : axis.Regular(180, 0, 6, name="dr_l0l1", label="dr between leading two leptons"),

            #"mlb_min" : axis.Regular(180, 0, 300, name="mlb_min",  label="min mass(b+l)"),
            #"mlb_max" : axis.Regular(180, 0, 1000, name="mlb_max",  label="max mass(b+l)"),

            "njets"   : axis.Regular(8, 0, 8, name="njets",   label="Jet multiplicity"),
            "nleps"   : axis.Regular(5, 0, 5, name="nleps",   label="Lep multiplicity"),
            "nbtagsl" : axis.Regular(4, 0, 4, name="nbtagsl", label="Loose btag multiplicity"),
            "nbtagsm" : axis.Regular(4, 0, 4, name="nbtagsm", label="Medium btag multiplicity"),

            "njets_counts"   : axis.Regular(30, 0, 30, name="njets_counts",   label="Jet multiplicity counts (central)"),
            "nleps_counts"   : axis.Regular(30, 0, 30, name="nleps_counts",   label="Lep multiplicity counts (central)"),

            "nfatjets"   : axis.Regular(8, 0, 8, name="nfatjets",   label="Fat jet multiplicity"),
            "njets_forward"   : axis.Regular(8, 0, 8, name="njets_forward",   label="Jet multiplicity (forward)"),
            "njets_tot"   : axis.Regular(8, 0, 8, name="njets_tot",   label="Jet multiplicity (central and forward)"),

            "fj0_pt"  : axis.Regular(180, 0, 2000, name="fj0_pt", label="fj0 pt"),
            "fj0_mass"  : axis.Regular(180, 0, 250, name="fj0_mass", label="fj0 mass"),
            "fj0_msoftdrop"  : axis.Regular(180, 0, 250, name="fj0_msoftdrop", label="fj0 softdrop mass"),
            "fj0_mparticlenet"  : axis.Regular(180, 0, 250, name="fj0_mparticlenet", label="fj0 particleNet mass"),
            "fj0_eta" : axis.Regular(180, -5, 5, name="fj0_eta", label="fj0 eta"),
            "fj0_phi" : axis.Regular(180, -3.1416, 3.1416, name="fj0_phi", label="j0 phi"),

            "fj0_pNetH4qvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetH4qvsQCD", label="fj0 pNet H4qvsQCD"),
            "fj0_pNetHbbvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetHbbvsQCD", label="fj0 pNet HbbvsQCD"),
            "fj0_pNetHccvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetHccvsQCD", label="fj0 pNet HccvsQCD"),
            "fj0_pNetQCD"     : axis.Regular(180, 0, 1, name="fj0_pNetQCD",    label="fj0 pNet QCD"),
            "fj0_pNetTvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetTvsQCD", label="fj0 pNet TvsQCD"),
            "fj0_pNetWvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetWvsQCD", label="fj0 pNet WvsQCD"),
            "fj0_pNetZvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetZvsQCD", label="fj0 pNet ZvsQCD"),

            "fj1_pt"  : axis.Regular(180, 0, 2000, name="fj1_pt", label="fj1 pt"),
            "fj1_mass"  : axis.Regular(180, 0, 250, name="fj1_mass", label="fj1 mass"),
            "fj1_msoftdrop"  : axis.Regular(180, 0, 250, name="fj1_msoftdrop", label="fj1 softdrop mass"),
            "fj1_mparticlenet"  : axis.Regular(180, 0, 250, name="fj1_mparticlenet", label="fj1 particleNet mass"),
            "fj1_eta" : axis.Regular(180, -5, 5, name="fj1_eta", label="fj1 eta"),
            "fj1_phi" : axis.Regular(180, -3.1416, 3.1416, name="fj1_phi", label="j0 phi"),

            "fj1_pNetH4qvsQCD": axis.Regular(180, 0, 1, name="fj1_pNetH4qvsQCD", label="fj1 pNet H4qvsQCD"),
            "fj1_pNetHbbvsQCD": axis.Regular(180, 0, 1, name="fj1_pNetHbbvsQCD", label="fj1 pNet HbbvsQCD"),
            "fj1_pNetHccvsQCD": axis.Regular(180, 0, 1, name="fj1_pNetHccvsQCD", label="fj1 pNet HccvsQCD"),
            "fj1_pNetQCD"     : axis.Regular(180, 0, 1, name="fj1_pNetQCD",    label="fj1 pNet QCD"),
            "fj1_pNetTvsQCD"  : axis.Regular(180, 0, 1, name="fj1_pNetTvsQCD", label="fj1 pNet TvsQCD"),
            "fj1_pNetWvsQCD"  : axis.Regular(180, 0, 1, name="fj1_pNetWvsQCD", label="fj1 pNet WvsQCD"),
            "fj1_pNetZvsQCD"  : axis.Regular(180, 0, 1, name="fj1_pNetZvsQCD", label="fj1 pNet ZvsQCD"),

            "j0central_pt"  : axis.Regular(180, 0, 250, name="j0central_pt", label="j0 pt (central jets)"), # Naming
            "j0central_eta" : axis.Regular(180, -5, 5, name="j0central_eta", label="j0 eta (central jets)"), # Naming
            "j0central_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0central_phi", label="j0 phi (central jets)"), # Naming


            "j0forward_pt"  : axis.Regular(180, 0, 150, name="j0forward_pt", label="j0 pt (forward jets)"),
            "j0forward_eta" : axis.Regular(180, -5, 5, name="j0forward_eta", label="j0 eta (forward jets)"),
            "j0forward_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0forward_phi", label="j0 phi (forward jets)"),

            "j0any_pt"  : axis.Regular(180, 0, 250, name="j0any_pt", label="j0 pt (all regular jets)"),
            "j0any_eta" : axis.Regular(180, -5, 5, name="j0any_eta", label="j0 eta (all regular jets)"),
            "j0any_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0any_phi", label="j0 phi (all regular jets)"),

            "dr_fj0l0" : axis.Regular(180, 0, 6, name="dr_fj0l0", label="dr between FJ and lepton"),
            "dr_j0fwdj1fwd" : axis.Regular(180, 0, 6, name="dr_j0fwdj1fwd", label="dr between leading two forward jets"),
            "dr_j0centj1cent" : axis.Regular(180, 0, 6, name="dr_j0centj1cent", label="dr between leading two central jets"),
            "dr_j0anyj1any" : axis.Regular(180, 0, 6, name="dr_j0anyj1any", label="dr between leading two jets"),

            "absdphi_j0fwdj1fwd"   : axis.Regular(180, 0, 3.1416, name="absdphi_j0fwdj1fwd", label="abs dphi between leading two forward jets"),
            "absdphi_j0centj1cent" : axis.Regular(180, 0, 3.1416, name="absdphi_j0centj1cent", label="abs dphi between leading two central jets"),
            "absdphi_j0anyj1any"   : axis.Regular(180, 0, 3.1416, name="absdphi_j0anyj1any", label="abs dphi between leading two jets"),

            "mass_j0centj1cent" : axis.Regular(180, 0, 250, name="mass_j0centj1cent", label="mjj of two leading (in pt) non-forward jets"),
            "mass_j0fwdj1fwd" : axis.Regular(180, 0, 2500, name="mass_j0fwdj1fwd", label="mjj of two leading (in pt) forward jets"),
            "mass_j0anyj1any" : axis.Regular(180, 0, 1500, name="mass_j0anyj1any", label="mjj of two leading (in pt) jets"),

            "mass_b0b1" : axis.Regular(180, 0, 250, name="mass_b0b1", label="mjj of two leading (pt) b jets"),

            "mass_bbscore0bbscore1" : axis.Regular(180, 0, 250, name="mass_bbscore0bbscore1", label="mjj of two leading (in score) loose b jets"),
            "mass_bmbscore0bmbscore1" : axis.Regular(180, 0, 250, name="mass_bmbscore0bmbscore1", label="mjj of two leading (in score) med b jets"),
            "bbscore0_bscore"  : axis.Regular(180, 0, 1, name="bbscore0_bscore", label="Btag score of b jet with highest btag score"),
            "bbscore1_bscore"  : axis.Regular(180, 0, 1, name="bbscore1_bscore", label="Btag score of b jet with second highest btag score"),

            "mass_jbscore0jbscore1" : axis.Regular(180, 0, 250, name="mass_jbscore0jbscore1", label="mjj of two leading (in score) central jets"),
            "jbscore0_bscore"  : axis.Regular(180, 0, 1, name="jbscore0_bscore", label="Btag score of central jet with highest btag score"),
            "jbscore1_bscore"  : axis.Regular(180, 0, 1, name="jbscore1_bscore", label="Btag score of central jet with second highest btag score"),

            "mjj_max_cent" : axis.Regular(180, 0, 250, name="mjj_max_cent", label="Leading mjj of pair of non-forward jets"),
            "mjj_max_fwd" : axis.Regular(180, 0, 2500, name="mjj_max_fwd", label="Leading mjj of pair of forward jets"),
            "mjj_max_any" : axis.Regular(180, 0, 1500, name="mjj_max_any", label="Leading mjj of pair of any (central or fwd) jets"),

            "jj_pairs_atmindr_mjj" : axis.Regular(180, 0, 1000, name="jj_pairs_atmindr_mjj", label="jj_pairs_atmindr_mjj"),

            "mjjjall_nearest_t" : axis.Regular(180, 0, 700, name="mjjjall_nearest_t", label="mjjj closest to top, considering all jets"),
            "mjjjcnt_nearest_t" : axis.Regular(180, 0, 700, name="mjjjcnt_nearest_t", label="mjjj closest to top, considering central jets"),

            "mjjjany" : axis.Regular(180, 0, 3000, name="mjjjany", label="mjjj of leading (in pt) three central or fwd jets"),
            "mjjjcnt" : axis.Regular(180, 0, 3000, name="mjjjcnt", label="mjjj of leading (in pt) three central jets"),
            "mjjjjany" : axis.Regular(180, 0, 4000, name="mjjjjany", label="mjjjj of leading (in pt) four central or fwd jets"),
            "mjjjjcnt" : axis.Regular(180, 0, 4000, name="mjjjjcnt", label="mjjjj of leading (in pt) four central jets"),

            "mljjjany" : axis.Regular(180, 0, 4000, name="mljjjany", label="mljjj of leading (in pt) lep and three central or fwd jets"),
            "mljjjcnt" : axis.Regular(180, 0, 4000, name="mljjjcnt", label="mljjj of leading (in pt) lep and three central jets"),
            "mljjjjany" : axis.Regular(180, 0, 4000, name="mljjjjany", label="mljjjj of leading (in pt) lep and four central or fwd jets"),
            "mljjjjcnt" : axis.Regular(180, 0, 4000, name="mljjjjcnt", label="mljjjj of leading (in pt) lep and four central jets"),

        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                hist.axis.StrCategory([], growth=True, name="systematic", label="systematic"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight", # Keeps track of sumw2
                name="Counts",
            )

        # Adding list accumulators for BDT output variables and weights
        if siphon_bdt_data:
            list_output_names = []
            for list_output_name in list_output_names:
                dout[list_output_name] = processor.list_accumulator([])

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

        # Set the booleans
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._skip_obj_systematics = skip_obj_systematics # Skip the JEC/JER/MET systematics (even if running with do_systematics on)
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        json_name = events.metadata["dataset"]

        isData       = self._samples[json_name]["isData"]
        histAxisName = events.namewithyear
        year         = events.year
        xsec         = events.xsec

        # FIXME Temp fix since only R2, maybe should specify in the input cfg
        is2022 = False
        is2023 = False

        # Era Needed for all samples
        if isData:
            era = self._samples[json_name]["era"]
        else:
            era = None


        # Get the dataset name (used for duplicate removal) and check to make sure it is an expected name
        # Get name for MC cases too, since "dataset" is passed to overlap removal function in all cases (though it's not actually used in the MC case)
        dataset = json_name.split('_')[0]
        if isData:
            datasets = ["SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG","Muon"]
            if dataset not in datasets:
                raise Exception(f"ERROR: Unexpected dataset name for data file: {dataset}")

        # Initialize objects
        ele     = events.Electron
        mu      = events.Muon
        jets    = events.Jet
        #met     = events.MET
        met     = events.PuppiMET
        fatjets = events.FatJet
        #fatjets = events.CorrFatJet
        #higgs   = events.Higgs

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(met.pt)


        ################### Lepton selection ####################

        # Get tight leptons for VVH selection, using mask from RDF
        # These now come from RDF output
        ele_vvh_t = ele
        mu_vvh_t  = mu

        l_vvh_t = ak.with_name(ak.concatenate([ele_vvh_t,mu_vvh_t],axis=1),'PtEtaPhiMCandidate')
        l_vvh_t = l_vvh_t[ak.argsort(l_vvh_t.pt, axis=-1,ascending=False)] # Sort by pt
        events["l_vvh_t"] = l_vvh_t

        l_vvh_t_padded = ak.pad_none(l_vvh_t, 4)
        l0 = l_vvh_t_padded[:,0]
        l1 = l_vvh_t_padded[:,1]
        l2 = l_vvh_t_padded[:,2]
        nleps = ak.num(l_vvh_t)


        ######### Normalization and weights ###########

        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        if not isData:
            if ak.any(events["LHEReweightingWeight"]):
                genw = events["LHEReweightingWeight"][:,60]
            else:
                genw = events["genWeight"]
            genw_raw = events["genWeight"]

            # Normalize to the weight from RDF
            weights_obj_base.add("norm",events.weight * genw/genw_raw)


        ######### The rest of the processor is inside this loop over systs that affect object kinematics  ###########

        obj_correction_systs = [
            #f"CMS_scale_j_{year}",
            #f"CMS_res_j_{year}",
            #f"CMS_scale_met_unclustered_energy_{year}",
        ]
        #obj_correction_systs = append_up_down_to_sys_base(obj_correction_systs)

        # If we're doing systematics and this isn't data, we will loop over the obj correction syst lst list
        if self._do_systematics and not isData and not self._skip_obj_systematics: obj_corr_syst_var_list = ["nominal"] + obj_correction_systs
        # Otherwise loop juse once, for nominal
        else: obj_corr_syst_var_list = ['nominal']

        # Loop over the list of systematic variations (that impact object kinematics) that we've constructed
        for obj_corr_syst_var in obj_corr_syst_var_list:
            # Make a copy of the base weights object, so that each time through the loop we do not double count systs
            # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics
            weights_obj_base_for_kinematic_syst = copy.deepcopy(weights_obj_base)


            #################### Jets ####################

            # Fat jets
            goodfatjets = fatjets[os_ec.is_good_fatjet(fatjets)]
            goodfatjets = os_ec.get_cleaned_collection(l_vvh_t,goodfatjets,drcut=0.8)

            # Clean with dr (though another option is to use jetIdx)
            cleanedJets = os_ec.get_cleaned_collection(l_vvh_t,jets) # Clean against leps
            cleanedJets = os_ec.get_cleaned_collection(goodfatjets,cleanedJets,drcut=0.8) # Clean against fat jets
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            # Jet Veto Maps
            # Removes events that have ANY jet in a specific eta-phi space (not required for Run 2)
            # Zero is passing the veto map, so Run 2 will be assigned an array of length events with all zeros
            veto_map_array = cor_ec.ApplyJetVetoMaps(cleanedJets, year) if (is2022 or is2023) else ak.zeros_like(met.pt)
            veto_map_mask = (veto_map_array == 0)

            # Selecting jets and cleaning them (already in RDF)
            goodJets = cleanedJets[(abs(cleanedJets.eta) <= 2.4)]
            goodJets_forward = cleanedJets[(abs(cleanedJets.eta) > 2.4)]

            # Count jets
            njets = ak.num(goodJets)
            njets_forward = ak.num(goodJets_forward)
            njets_tot = njets + njets_forward
            nfatjets = ak.num(goodfatjets)
            ht = ak.sum(goodJets.pt,axis=-1)

            goodJets_ptordered = goodJets[ak.argsort(goodJets.pt,axis=-1,ascending=False)]
            goodJets_ptordered_padded = ak.pad_none(goodJets_ptordered, 4)
            j0 = goodJets_ptordered_padded[:,0]
            j1 = goodJets_ptordered_padded[:,1]
            j2 = goodJets_ptordered_padded[:,2]
            j3 = goodJets_ptordered_padded[:,3]

            goodJets_forward_ptordered = goodJets_forward[ak.argsort(goodJets_forward.pt,axis=-1,ascending=False)]
            goodJets_forward_ptordered_padded = ak.pad_none(goodJets_forward_ptordered, 2)
            j0forward = goodJets_forward_ptordered_padded[:,0]
            j1forward = goodJets_forward_ptordered_padded[:,1]

            goodJetsCentFwd = ak.with_name(ak.concatenate([goodJets,goodJets_forward],axis=1),'PtEtaPhiMLorentzVector')
            goodJetsCentFwd_ptordered = goodJetsCentFwd[ak.argsort(goodJetsCentFwd.pt,axis=-1,ascending=False)]
            goodJetsCentFwd_ptordered_padded = ak.pad_none(goodJetsCentFwd_ptordered, 4)
            j0any = goodJetsCentFwd_ptordered_padded[:,0]
            j1any = goodJetsCentFwd_ptordered_padded[:,1]
            j2any = goodJetsCentFwd_ptordered_padded[:,2]
            j3any = goodJetsCentFwd_ptordered_padded[:,3]

            goodfatjets_ptordered = goodfatjets[ak.argsort(goodfatjets.pt,axis=-1,ascending=False)]
            goodfatjets_ptordered_padded = ak.pad_none(goodfatjets_ptordered, 2)
            fj0 = goodfatjets_ptordered_padded[:,0]
            fj1 = goodfatjets_ptordered_padded[:,1]

            scalarptsum_jetCentFwd = ak.sum(goodJetsCentFwd.pt,axis=-1)
            scalarptsum_jetCent = ak.sum(goodJets.pt,axis=-1)
            scalarptsum_jetFwd = ak.sum(goodJets_forward.pt,axis=-1)

            mjjjany = ak.where(njets_tot>=3, (j0any+j1any+j2any).mass, 0)
            mjjjcnt = ak.where(njets>=3, (j0+j1+j2).mass, 0)
            mjjjjany = ak.where(njets_tot>=4, (j0any+j1any+j2any+j3any).mass, 0)
            mjjjjcnt = ak.where(njets>=4, (j0+j1+j2+j3).mass, 0)

            mljjjany = ak.where(njets_tot>=3, (l0 + j0any+j1any+j2any).mass, 0)
            mljjjcnt = ak.where(njets>=3, (l0 + j0+j1+j2).mass, 0)
            mljjjjany = ak.where(njets_tot>=4, (l0 + j0any+j1any+j2any+j3any).mass, 0)
            mljjjjcnt = ak.where(njets>=4, (l0 + j0+j1+j2+j3).mass, 0)



            ### Btag WPs ###
            # Eventually we should probably just take the btag jet collection from the RDF output
            # For now handle it with a hard-to-read series of ak.where
            btagwpl = events.nom
            btagwpm = events.nom
            btagwpl = ak.where(events.year=="2018",get_tc_param("btag_wp_loose_UL18"),btagwpl)
            btagwpm = ak.where(events.year=="2018",get_tc_param("btag_wp_loose_UL18"),btagwpm)
            btagwpl = ak.where(events.year=="2017",get_tc_param("btag_wp_loose_UL17"),btagwpl)
            btagwpm = ak.where(events.year=="2017",get_tc_param("btag_wp_loose_UL17"),btagwpm)
            btagwpl = ak.where(events.year=="2016postVFP",get_tc_param("btag_wp_loose_UL16"),btagwpl)
            btagwpm = ak.where(events.year=="2016postVFP",get_tc_param("btag_wp_loose_UL16"),btagwpm)
            btagwpl = ak.where(events.year=="2016preVFP",get_tc_param("btag_wp_loose_UL16APV"),btagwpl)
            btagwpm = ak.where(events.year=="2016preVFP",get_tc_param("btag_wp_loose_UL16APV"),btagwpm)

            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)

            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])


            ######### Masks we need for the selection ##########

            # Pass trigger mask
            era_for_trg_check = era
            if not (is2022 or is2023):
                # Era not used for R2
                era_for_trg_check = None
            #pass_trg = es_tc.trg_pass_no_overlap(events,isData,dataset,str(year),dataset_dict=es_ec.dataset_dict,exclude_dict=es_ec.exclude_dict,era=era_for_trg_check)
            #pass_trg = (pass_trg & es_ec.trg_matching(events,year))

            # b jet masks
            bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
            bmask_exactly0loose = (nbtagsl==0) # Used for 4l WWZ SR
            bmask_exactly0med = (nbtagsm==0) # Used for 3l CR and 2los Z CR
            bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
            bmask_exactly2med = (nbtagsm==2) # Used for CRtt
            bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR
            bmask_atmost2med  = (nbtagsm< 3) # Used to make 2lss mutually exclusive from tttt enriched
            bmask_atleast3med = (nbtagsm>=3) # Used for tttt enriched
            bmask_atleast1med = (nbtagsm>=1)
            bmask_atleast1loose = (nbtagsl>=1)
            bmask_atleast2loose = (nbtagsl>=2)


            ######### Get variables we haven't already calculated #########

            # Replace with 0 when there are not a pair of jets
            mjj_tmp = (j0+j1).mass
            mass_j0centj1cent = ak.where(njets>1,mjj_tmp,0)

            j0forward_eta = ak.where(njets_forward>0,j0forward.eta,0)

            j0any_pt = ak.where(njets_tot>0,j0any.pt,0)

            mass_j0anyj1any = ak.where(njets_tot>1,(j0any+j1any).mass,0)

            mass_j0fwdj1fwd = ak.where(njets_forward>1,(j0forward+j1forward).mass,0)


            # Find the mjj of the pair of jets (central + fwd) that have the min delta R
            jj_pairs = ak.combinations(goodJetsCentFwd_ptordered_padded, 2, fields=["j0", "j1"] )
            jj_pairs_dr = jj_pairs.j0.delta_r(jj_pairs.j1)
            jj_pairs_idx_mindr = ak.argmin(jj_pairs_dr,axis=1,keepdims=True)
            jj_pairs_atmindr = jj_pairs[jj_pairs_idx_mindr]
            jj_pairs_atmindr_mjj = (jj_pairs_atmindr.j0 + jj_pairs_atmindr.j1).mass
            jj_pairs_atmindr_mjj = ak.flatten(ak.fill_none(jj_pairs_atmindr_mjj,-999)) # Replace Nones, flatten (so e.g. [[None],[x],[y]] -> [-999,x,y])

            # Find jet triplets clost to top mass
            jetall_triplets = ak.combinations(goodJetsCentFwd_ptordered_padded, 3, fields=["j0", "j1", "j2"] )
            jetcnt_triplets = ak.combinations(goodJets_ptordered_padded,        3, fields=["j0", "j1", "j2"] )
            jjjall_4vec = jetall_triplets.j0 + jetall_triplets.j1 + jetall_triplets.j2
            jjjcnt_4vec = jetcnt_triplets.j0 + jetcnt_triplets.j1 + jetcnt_triplets.j2
            tpeak_jall_idx = ak.argmin(abs(jjjall_4vec.mass - 273),keepdims=True,axis=1)
            tpeak_jcnt_idx = ak.argmin(abs(jjjcnt_4vec.mass - 273),keepdims=True,axis=1)
            mjjjall_nearest_t = ak.fill_none(ak.flatten(jjjall_4vec[tpeak_jall_idx].mass),0)
            mjjjcnt_nearest_t = ak.fill_none(ak.flatten(jjjcnt_4vec[tpeak_jcnt_idx].mass),0)

            l0_pt = l0.pt
            l0_eta = l0.eta
            l1_pt = l1.pt
            l1_eta = l1.eta
            l2_pt = l2.pt
            l2_eta = l2.eta
            j0central_pt = j0.pt
            j0central_eta = j0.eta
            j0central_phi = j0.phi
            mass_l0l1 = (l0+l1).mass
            dr_l0l1 = l0.delta_r(l1)
            scalarptsum_lep = ak.sum(l_vvh_t.pt,axis=-1)
            scalarptsum_lepmet = scalarptsum_lep + met.pt
            scalarptsum_lepmetFJ = scalarptsum_lep + met.pt + fj0.pt
            scalarptsum_lepmetFJ10 = scalarptsum_lep + met.pt + fj0.pt + fj1.pt
            scalarptsum_lepmetalljets = scalarptsum_lep + met.pt + scalarptsum_jetCentFwd
            scalarptsum_lepmetcentjets = scalarptsum_lep + met.pt + scalarptsum_jetCent
            scalarptsum_lepmetfwdjets = scalarptsum_lep + met.pt + scalarptsum_jetFwd

            # lb pairs (i.e. always one lep, one bjet)
            bjets = goodJets[isBtagJetsLoose]
            bjetsm = goodJets[isBtagJetsMedium]
            lb_pairs = ak.cartesian({"l":l_vvh_t,"j":bjets})
            mlb_min = ak.min((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)
            mlb_max = ak.max((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)

            bjets_ptordered = bjets[ak.argsort(bjets.pt,axis=-1,ascending=False)]
            bjets_ptordered_padded = ak.pad_none(bjets_ptordered, 2)
            b0 = bjets_ptordered_padded[:,0]
            b1 = bjets_ptordered_padded[:,1]
            mass_b0b1_tmp = (b0+b1).mass
            mass_b0b1 = ak.where(nbtagsl>1,mass_b0b1_tmp,0)

            # Variables related to leading b jet score of b jets
            bjets_bscoreordered = bjets[ak.argsort(bjets.btagDeepFlavB,axis=-1,ascending=False)]
            bjets_bscoreordered_padded = ak.pad_none(bjets_bscoreordered, 2)
            bbscore0 = bjets_bscoreordered_padded[:,0]
            bbscore1 = bjets_bscoreordered_padded[:,1]
            mass_bbscore0bbscore1 = ak.fill_none((bbscore0+bbscore1).mass,0)
            bbscore0_bscore = ak.fill_none(bbscore0.btagDeepFlavB,0)
            bbscore1_bscore = ak.fill_none(bbscore1.btagDeepFlavB,0)

            # Variables related to leading b jet score of med b jets
            bjetsm_bscoreordered = bjetsm[ak.argsort(bjetsm.btagDeepFlavB,axis=-1,ascending=False)]
            bjetsm_bscoreordered_padded = ak.pad_none(bjetsm_bscoreordered, 2)
            bmbscore0 = bjetsm_bscoreordered_padded[:,0]
            bmbscore1 = bjetsm_bscoreordered_padded[:,1]
            mass_bmbscore0bmbscore1 = ak.fill_none((bmbscore0+bmbscore1).mass,0)

            # Variables related to leading b jet score of central jets
            centraljets_bscoreordered = goodJets_ptordered_padded[ak.argsort(goodJets_ptordered_padded.btagDeepFlavB,axis=-1,ascending=False)]
            jbscore0 = centraljets_bscoreordered[:,0]
            jbscore1 = centraljets_bscoreordered[:,1]
            mass_jbscore0jbscore1 = ak.fill_none((jbscore0+jbscore1).mass,0)
            jbscore0_bscore = ak.fill_none(jbscore0.btagDeepFlavB,0)
            jbscore1_bscore = ak.fill_none(jbscore1.btagDeepFlavB,0)

            # Mjj max from any jets
            jjCentFwd_pairs = ak.combinations( goodJetsCentFwd_ptordered_padded, 2, fields=["j0", "j1"] )
            mjj_max_any  = ak.fill_none(ak.max((jjCentFwd_pairs.j0 + jjCentFwd_pairs.j1).mass,axis=-1),0)

            # Mjj max from cent jets
            jjCent_pairs = ak.combinations(goodJets_ptordered_padded, 2, fields=["j0", "j1"] )
            mjj_max_cent = ak.fill_none(ak.max((jjCent_pairs.j0 + jjCent_pairs.j1).mass,axis=-1),0)

            # Mjj max from forward jets
            jjFwd_pairs = ak.combinations(goodJets_forward_ptordered_padded, 2, fields=["j0", "j1"] )
            mjj_max_fwd = ak.fill_none(ak.max((jjFwd_pairs.j0 + jjFwd_pairs.j1).mass,axis=-1),0)

            ### TMP!!! These are not in R3, so just use particleNet_QCD for all for now so we don't crash ###
            fj0_pNetH4qvsQCD = fj0.particleNet_QCD
            fj0_pNetHbbvsQCD = fj0.particleNet_QCD
            fj0_pNetHccvsQCD = fj0.particleNet_QCD
            fj0_pNetQCD      = fj0.particleNet_QCD
            fj0_pNetTvsQCD   = fj0.particleNet_QCD
            fj0_pNetWvsQCD   = fj0.particleNet_QCD
            fj0_pNetZvsQCD   = fj0.particleNet_QCD
            fj0_mparticlenet = fj0.particleNet_QCD

            fj1_pNetH4qvsQCD = fj1.particleNet_QCD
            fj1_pNetHbbvsQCD = fj1.particleNet_QCD
            fj1_pNetHccvsQCD = fj1.particleNet_QCD
            fj1_pNetQCD      = fj1.particleNet_QCD
            fj1_pNetTvsQCD   = fj1.particleNet_QCD
            fj1_pNetWvsQCD   = fj1.particleNet_QCD
            fj1_pNetZvsQCD   = fj1.particleNet_QCD
            fj1_mparticlenet = fj1.particleNet_QCD

            # Only for R2
            #fj0_pNetH4qvsQCD = fj0.particleNet_H4qvsQCD
            #fj0_pNetHbbvsQCD = fj0.particleNet_HbbvsQCD
            #fj0_pNetHccvsQCD = fj0.particleNet_HccvsQCD
            #fj0_pNetQCD      = fj0.particleNet_QCD
            #fj0_pNetTvsQCD   = fj0.particleNet_TvsQCD
            #fj0_pNetWvsQCD   = fj0.particleNet_WvsQCD
            #fj0_pNetZvsQCD   = fj0.particleNet_ZvsQCD
            #fj0_mparticlenet = fj0.particleNet_mass
            #fj1_pNetH4qvsQCD = fj1.particleNet_H4qvsQCD
            #fj1_pNetHbbvsQCD = fj1.particleNet_HbbvsQCD
            #fj1_pNetHccvsQCD = fj1.particleNet_HccvsQCD
            #fj1_pNetQCD      = fj1.particleNet_QCD
            #fj1_pNetTvsQCD   = fj1.particleNet_TvsQCD
            #fj1_pNetWvsQCD   = fj1.particleNet_WvsQCD
            #fj1_pNetZvsQCD   = fj1.particleNet_ZvsQCD
            #fj1_mparticlenet = fj1.particleNet_mass
            ###


            ### Put the variables we'll plot into a dictionary for easy access later ###
            dense_variables_dict = {
                "met" : met.pt,
                "metphi" : met.phi,
                "scalarptsum_lep" : scalarptsum_lep,
                "scalarptsum_jetCentFwd" : scalarptsum_jetCentFwd,
                "scalarptsum_jetCent" : scalarptsum_jetCent,
                "scalarptsum_jetFwd" : scalarptsum_jetFwd,
                "scalarptsum_lepmet" : scalarptsum_lepmet,
                "scalarptsum_lepmetFJ" : scalarptsum_lepmetFJ,
                "scalarptsum_lepmetFJ10" : scalarptsum_lepmetFJ10,
                "scalarptsum_lepmetalljets" : scalarptsum_lepmetalljets,
                "scalarptsum_lepmetcentjets" : scalarptsum_lepmetcentjets,
                "scalarptsum_lepmetfwdjets" : scalarptsum_lepmetfwdjets,
                "l0_pt" : l0_pt,
                "l0_eta" : l0_eta,
                "l1_pt" : l1_pt,
                "l1_eta" : l1_eta,
                "l2_pt" : l2_pt,
                "l2_eta" : l2_eta,
                "mass_l0l1" : mass_l0l1,
                "dr_l0l1" : dr_l0l1,

                "j0central_pt" : j0central_pt,
                "j0central_eta" : j0central_eta,
                "j0central_phi" : j0central_phi,

                "j0forward_pt" : j0forward.pt,
                "j0forward_eta" : j0forward_eta,
                "j0forward_phi" : j0forward.phi,

                "j0any_pt" : j0any_pt,
                "j0any_eta" : j0any.eta,
                "j0any_phi" : j0any.phi,

                "nleps" : nleps,
                "njets" : njets,
                "nbtagsl" : nbtagsl,

                "nleps_counts" : nleps,
                "njets_counts" : njets,
                "nbtagsl_counts" : nbtagsl,

                "nbtagsm" : nbtagsm,
                "nbtagsl" : nbtagsl,

                "nfatjets" : nfatjets,
                "njets_forward" : njets_forward,
                "njets_tot" : njets_tot,
                "fj0_pt" : fj0.pt,
                "fj0_mass" : fj0.mass,
                "fj0_msoftdrop" : fj0.msoftdrop,
                "fj0_eta" : fj0.eta,
                "fj0_phi" : fj0.phi,

                "fj1_pt" : fj1.pt,
                "fj1_mass" : fj1.mass,
                "fj1_msoftdrop" : fj1.msoftdrop,
                "fj1_eta" : fj1.eta,
                "fj1_phi" : fj1.phi,

                "j0_pt" : j0.pt,
                "j0_eta" : j0.eta,
                "j0_phi" : j0.phi,

                "dr_fj0l0" : fj0.delta_r(l0),
                "dr_j0fwdj1fwd" : j0forward.delta_r(j1forward),
                "dr_j0centj1cent" : j0.delta_r(j1),
                "dr_j0anyj1any" : j0any.delta_r(j1any),
                "absdphi_j0fwdj1fwd"   : abs(j0forward.delta_phi(j1forward)),
                "absdphi_j0centj1cent" : abs(j0.delta_phi(j1)),
                "absdphi_j0anyj1any"   : abs(j0any.delta_phi(j1any)),

                "mass_j0centj1cent" : mass_j0centj1cent,
                "mass_j0fwdj1fwd" : mass_j0fwdj1fwd,
                "mass_j0anyj1any" : mass_j0anyj1any,

                "mass_b0b1" : mass_b0b1,

                "fj0_pNetH4qvsQCD" : fj0_pNetH4qvsQCD,
                "fj0_pNetHbbvsQCD" : fj0_pNetHbbvsQCD,
                "fj0_pNetHccvsQCD" : fj0_pNetHccvsQCD,
                "fj0_pNetQCD"      : fj0_pNetQCD,
                "fj0_pNetTvsQCD"   : fj0_pNetTvsQCD,
                "fj0_pNetWvsQCD"   : fj0_pNetWvsQCD,
                "fj0_pNetZvsQCD"   : fj0_pNetZvsQCD,
                "fj0_mparticlenet" : fj0_mparticlenet,

                "fj1_pNetH4qvsQCD" : fj1_pNetH4qvsQCD,
                "fj1_pNetHbbvsQCD" : fj1_pNetHbbvsQCD,
                "fj1_pNetHccvsQCD" : fj1_pNetHccvsQCD,
                "fj1_pNetQCD"      : fj1_pNetQCD,
                "fj1_pNetTvsQCD"   : fj1_pNetTvsQCD,
                "fj1_pNetWvsQCD"   : fj1_pNetWvsQCD,
                "fj1_pNetZvsQCD"   : fj1_pNetZvsQCD,
                "fj1_mparticlenet" : fj1_mparticlenet,

                "jj_pairs_atmindr_mjj" : jj_pairs_atmindr_mjj,

                "bbscore0_bscore" : bbscore0_bscore,
                "bbscore1_bscore" : bbscore1_bscore,
                "mass_bbscore0bbscore1" : mass_bbscore0bbscore1,
                "mass_bmbscore0bmbscore1" : mass_bmbscore0bmbscore1,

                "jbscore0_bscore" : jbscore0_bscore,
                "jbscore1_bscore" : jbscore1_bscore,
                "mass_jbscore0jbscore1" : mass_jbscore0jbscore1,

                "mjj_max_any" : mjj_max_any,
                "mjj_max_cent" : mjj_max_cent,
                "mjj_max_fwd" : mjj_max_fwd,

                "mjjjall_nearest_t": mjjjall_nearest_t,
                "mjjjcnt_nearest_t": mjjjcnt_nearest_t,

                "mjjjany" : mjjjany,
                "mjjjcnt" : mjjjcnt,
                "mjjjjany" : mjjjjany,
                "mjjjjcnt" : mjjjjcnt,

                "mljjjany" : mljjjany,
                "mljjjcnt" : mljjjcnt,
                "mljjjjany" : mljjjjany,
                "mljjjjcnt" : mljjjjcnt,

            }


            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint64')

            # Form some other useful masks for SRs

            mask_exactly1lep_exactly2fj = (nleps==1) & (nfatjets==2)
            mask_exactly1lep_exactly1fj = (nleps==1) & (nfatjets==1)
            mask_presel = mask_exactly1lep_exactly1fj & (scalarptsum_lepmet > 775)

            mask_preselHFJ = mask_presel & (fj0_mparticlenet >  100.) & (fj0_mparticlenet <= 150.)
            mask_preselVFJ = mask_presel & (fj0_mparticlenet <= 100.) & (fj0_mparticlenet > 65)
            mask_HFJ = (fj0_mparticlenet >  100.) & (fj0_mparticlenet <= 150.)
            mask_VFJ = (fj0_mparticlenet <= 100.) & (fj0_mparticlenet > 65)

            mask_preselHFJTag = mask_preselHFJ & (fj0_pNetHbbvsQCD > 0.98) & (fj0_pNetTvsQCD < 0.5) & (fj0_pNetWvsQCD < 0.5)
            mask_preselVFJTag = mask_preselVFJ & (fj0_pNetWvsQCD > 0.95) & (fj0_pNetTvsQCD < 0.5)
            mask_HFJTagHbb = (fj0_pNetHbbvsQCD > 0.98)
            mask_HFJtag = (fj0_pNetHbbvsQCD > 0.98) & (fj0_pNetTvsQCD < 0.5) & (fj0_pNetWvsQCD < 0.5)
            mask_VFJtag = (fj0_pNetWvsQCD > 0.95) & (fj0_pNetTvsQCD < 0.5)

            ### Pre selections ###
            selections.add("all_events", (veto_map_mask | (~veto_map_mask))) # All events.. this logic is a bit roundabout to just get an array of True

            ### 1lep + 1FJ ###
            selections.add("exactly1lep_exactly1fj" , mask_exactly1lep_exactly1fj)
            selections.add("presel", mask_presel)
            # HFJ selections
            selections.add("preselHFJ", mask_preselHFJ)
            selections.add("preselHFJTag",mask_preselHFJTag)
            selections.add("preselHFJTag_mjj115", mask_preselHFJTag & (mass_j0centj1cent < 115))
            # VFJ selections
            selections.add("preselVFJ", mask_preselVFJ)
            selections.add("preselVFJTag",                                  mask_preselVFJTag)
            selections.add("preselVFJTag_mjjcent75to150",                   mask_preselVFJTag & (mass_j0centj1cent>75) & (mass_j0centj1cent<150))
            selections.add("preselVFJTag_mjjcent75to150_mbb75to150",        mask_preselVFJTag & (mass_j0centj1cent>75) & (mass_j0centj1cent<150) & (mass_b0b1>75) & (mass_b0b1<150))
            selections.add("preselVFJTag_mjjcent75to150_mbb75to150_mvqq75p",mask_preselVFJTag & (mass_j0centj1cent>75) & (mass_j0centj1cent<150) & (mass_b0b1>75) & (mass_b0b1<150) & (jj_pairs_atmindr_mjj > 75))

            ### 1lep + 2FJ ###
            selections.add("exactly1lep_exactly2fj" ,                          mask_exactly1lep_exactly2fj)
            selections.add("exactly1lep_exactly2fj_lepmet600" ,                mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600))
            selections.add("exactly1lep_exactly2fj_lepmet600_VFJ" ,            mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600) & mask_VFJ)
            selections.add("exactly1lep_exactly2fj_lepmet600_VFJtag" ,         mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600) & mask_VFJ & mask_VFJtag)
            selections.add("exactly1lep_exactly2fj_lepmet600_VFJtag_njcent0" , mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600) & mask_VFJ & mask_VFJtag & (njets==0))
            selections.add("exactly1lep_exactly2fj_lepmet600_HFJ" ,            mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600) & mask_HFJ)
            selections.add("exactly1lep_exactly2fj_lepmet600_HFJtagZ" ,        mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600) & mask_HFJ & (fj0_pNetZvsQCD>0.5))
            selections.add("exactly1lep_exactly2fj_lepmet600_HFJtagZ_njcent0" ,mask_exactly1lep_exactly2fj & (scalarptsum_lepmet>600) & mask_HFJ & (fj0_pNetZvsQCD>0.5) & (njets==0))

            selections.add("exactly1lep_l40_noloosel" , (nleps==1)    & (l0.pt>40))
            ## Test
            selections.add("exactly2fj_l40_noloosel"  , (nfatjets==2) & (l0.pt>40))
            selections.add("exactly1lep_l40_noloosel" , (nleps==1)    & (l0.pt>40))
            selections.add("exactly1lep_exactly2fj_l40_noloosel"  , (nleps==1) & (nfatjets==2) & (l0.pt>40))
            ##

            ### 2lOS + 1FJ ###
            os_mask = l0.pdgId*l1.pdgId<0
            ss_mask = l0.pdgId*l1.pdgId>0
            mask_exactly2lepOS = (nleps==2) & os_mask
            mask_exactly2lepSS = (nleps==2) & ss_mask
            mask_exactly2lepSS_exactly1fj = mask_exactly2lepSS & (nfatjets==1)
            mask_exactly2lepOS_exactly1fj = mask_exactly2lepOS & (nfatjets==1)
            mask_exactly2lepOS_exactly2fj = mask_exactly2lepOS & (nfatjets==2)
            selections.add("exactly2lepOS"                                    , mask_exactly2lepOS)
            selections.add("exactly2lepOS_exactly1fj"                         , mask_exactly2lepOS_exactly1fj)
            selections.add("exactly2lepOS_exactly1fj_HFJ"                     , mask_exactly2lepOS_exactly1fj & mask_HFJ)
            selections.add("exactly2lepOS_exactly1fj_HFJtag"                  , mask_exactly2lepOS_exactly1fj & mask_HFJ & mask_HFJTagHbb)
            selections.add("exactly2lepOS_exactly1fj_HFJtag_lepmetjetf800"    , mask_exactly2lepOS_exactly1fj & mask_HFJ & mask_HFJTagHbb & (scalarptsum_lepmetfwdjets>800))
            selections.add("exactly2lepOS_exactly1fj_VFJ"                     , mask_exactly2lepOS_exactly1fj & mask_VFJ)
            selections.add("exactly2lepOS_exactly1fj_VFJ_met100"              , mask_exactly2lepOS_exactly1fj & mask_VFJ & (met.pt>100))
            selections.add("exactly2lepOS_exactly1fj_VFJ_met100_lepmetjetf700", mask_exactly2lepOS_exactly1fj & mask_VFJ & (met.pt>100) & (scalarptsum_lepmetfwdjets>700))



            cat_dict = {
                "lep_chan_lst" : [

                    "all_events",
                    #"exactly1lep",

                    ### 1lep 1FJ ###
                    "exactly1lep_exactly1fj",
                    "presel",
                    "preselHFJ",
                    "preselHFJTag",
                    "preselHFJTag_mjj115",
                    "preselVFJ",
                    "preselVFJTag",
                    "preselVFJTag_mjjcent75to150",
                    "preselVFJTag_mjjcent75to150_mbb75to150",
                    "preselVFJTag_mjjcent75to150_mbb75to150_mvqq75p",

                    ### 1lep+2FJ ###
                    "exactly1lep_exactly2fj",
                    "exactly1lep_exactly2fj_lepmet600",
                    "exactly1lep_exactly2fj_lepmet600_VFJ",
                    "exactly1lep_exactly2fj_lepmet600_VFJtag",
                    "exactly1lep_exactly2fj_lepmet600_VFJtag_njcent0",
                    "exactly1lep_exactly2fj_lepmet600_HFJ",
                    "exactly1lep_exactly2fj_lepmet600_HFJtagZ",
                    "exactly1lep_exactly2fj_lepmet600_HFJtagZ_njcent0",
                    ## Test
                    "exactly2fj_l40_noloosel",
                    "exactly1lep_l40_noloosel",
                    "exactly1lep_exactly2fj_l40_noloosel",
                    ##

                    ### 2lOS 1FJ ###
                    "exactly2lepOS",
                    "exactly2lepOS_exactly1fj",
                    "exactly2lepOS_exactly1fj_HFJ",
                    "exactly2lepOS_exactly1fj_HFJtag",
                    "exactly2lepOS_exactly1fj_HFJtag_lepmetjetf800",
                    #"exactly2lepOS_exactly1fj_VFJ",
                    #"exactly2lepOS_exactly1fj_VFJ_met100",
                    #"exactly2lepOS_exactly1fj_VFJ_met100_lepmetjetf700",

                ]
            }


            ######### Fill histos #########

            exclude_var_dict = {} # Any particular ones to skip

            wgt_correction_syst_lst = []

            # Set up the list of weight fluctuations to loop over
            # For now the syst do not depend on the category, so we can figure this out outside of the filling loop
            wgt_var_lst = ["nominal"]
            if self._do_systematics:
                if not isData:
                    if (obj_corr_syst_var != "nominal"):
                        # In this case, we are dealing with systs that change the kinematics of the objs (e.g. JES)
                        # So we don't want to loop over up/down weight variations here
                        wgt_var_lst = [obj_corr_syst_var]
                    else:
                        # Otherwise we want to loop over the up/down weight variations
                        wgt_var_lst = wgt_var_lst + wgt_correction_syst_lst



            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in dense_variables_dict.items():
                if dense_axis_name not in self._hist_lst:
                    #print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                    continue

                # Loop over weight fluctuations
                for wgt_fluct in wgt_var_lst:

                    # Get the appropriate weight fluctuation
                    if (wgt_fluct == "nominal") or (wgt_fluct in obj_corr_syst_var_list):
                        # In the case of "nominal", no weight systematic variation is used
                        weight = weights_obj_base_for_kinematic_syst.weight(None)
                    else:
                        # Otherwise get the weight from the Weights object
                        weight = weights_obj_base_for_kinematic_syst.weight(wgt_fluct)


                    # Loop over categories
                    for sr_cat in cat_dict["lep_chan_lst"]:

                        # Skip filling if this variable is not relevant for this selection
                        if (dense_axis_name in exclude_var_dict) and (sr_cat in exclude_var_dict[dense_axis_name]): continue

                        # If this is a counts hist, forget the weights and just fill with unit weights
                        if dense_axis_name.endswith("_counts"): weight = events.nom

                        # Make the cuts mask
                        cuts_lst = [sr_cat]
                        all_cuts_mask = selections.all(*cuts_lst)

                        # Print info about the events
                        #import sys
                        #run = events.run[all_cuts_mask]
                        #luminosityBlock = events.luminosityBlock[all_cuts_mask]
                        #event = events.event[all_cuts_mask]
                        #w = weight[all_cuts_mask]
                        #if dense_axis_name == "njets":
                        #    print("\nSTARTPRINT")
                        #    for i,j in enumerate(w):
                        #        out_str = f"PRINTTAG {i} {dense_axis_name} {year} {sr_cat} {event[i]} {run[i]} {luminosityBlock[i]} {w[i]}"
                        #        print(out_str,file=sys.stderr,flush=True)
                        #    print("ENDPRINT\n")
                        #print("\ndense_axis_name",dense_axis_name)
                        #print("sr_cat",sr_cat)
                        #print("dense_axis_vals[all_cuts_mask]",dense_axis_vals[all_cuts_mask])
                        #print("end")

                        # Fill the histos
                        axes_fill_info_dict = {
                            dense_axis_name : ak.fill_none(dense_axis_vals[all_cuts_mask],0), # Don't like this fill_none
                            "weight"        : ak.fill_none(weight[all_cuts_mask],0),          # Don't like this fill_none
                            #"weight"        : ak.fill_none(events.weight[all_cuts_mask],0),          # Don't like this fill_none
                            "process"       : histAxisName[all_cuts_mask],
                            "category"      : sr_cat,
                            "systematic"    : wgt_fluct,
                        }
                        self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

        return self.accumulator

    def postprocess(self, accumulator):
        return accumulator
