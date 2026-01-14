from hist import axis
from coffea.nanoevents.methods import vector
import awkward as ak
ak.behavior.update(vector.behavior)
import numpy as np


##################################################
## Define extra objects
##################################################
def get_leading_jet(jets):
    jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    return ak.firsts(jets_sorted)

def get_max_score_jet(jets, score_field):
    """Get the jet with the maximum value of score_field"""
    scores = getattr(jets, score_field)
    jets_sorted = jets[ak.argsort(scores, axis=1, ascending=False)]
    return ak.firsts(jets_sorted)

def get_padded_jets(jets, n, sort_field="pt"):
    """Sort jets by field and pad to n jets"""
    jets_sorted = jets[ak.argsort(getattr(jets, sort_field), axis=1, ascending=False)]
    return ak.pad_none(jets_sorted, n)

def get_cleaned_collection(obj_to_clean, obj_to_avoid, drcut=0.4):
    """
    Remove objects from obj_to_clean that overlap with any object in obj_to_avoid.
    Based on ewkcoffea/modules/objects_wwz.py

    If obj_to_avoid is empty/None for an event, all objects are kept (dr=None -> mask=True).
    """
    _, dr = obj_to_clean.nearest(obj_to_avoid, return_metric=True)
    mask = ak.fill_none(dr > drcut, True)
    return obj_to_clean[mask]


objects_config = {
    # Max Hbb score AK8
    "maxHbbAK8": lambda events: ak.with_name(
        get_max_score_jet(events.fatjet, "HbbScore"),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Max Wqq score AK8
    "maxWqqAK8": lambda events: ak.with_name(
        get_max_score_jet(events.fatjet, "WqqScore"),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Leading fatjet (for 1-FJ category)
    "leading_fj": lambda events: ak.with_name(
        get_leading_jet(events.fatjet),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Central jets (|eta| <= 2.4) - raw, before fatjet overlap removal
    "jets_central_raw": lambda events: ak.with_name(
        events.jet[(abs(events.jet.eta) <= 2.4)],
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Forward jets (|eta| > 2.4) - raw, before fatjet overlap removal
    "jets_forward_raw": lambda events: ak.with_name(
        events.jet[(abs(events.jet.eta) > 2.4)],
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
}

derived_objects_config = {
    # Padded fatjets (fj0, fj1)
    "fatjets_padded": lambda events, objects: ak.pad_none(
        events.fatjet[ak.argsort(events.fatjet.pt, axis=1, ascending=False)], 2
    ),
    # Central jets cleaned of leading fatjet overlap (dR > 0.8)
    "jets_central": lambda events, objects: ak.with_name(
        get_cleaned_collection(objects["jets_central_raw"], ak.singletons(objects["leading_fj"]), drcut=0.8),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Forward jets cleaned of leading fatjet overlap (dR > 0.8)
    "jets_forward": lambda events, objects: ak.with_name(
        get_cleaned_collection(objects["jets_forward_raw"], ak.singletons(objects["leading_fj"]), drcut=0.8),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
}

# Second level derived objects (depend on first level derived objects)
derived_objects_config_level2 = {
    # Padded central jets (j0, j1, j2, j3)
    "jets_central_padded": lambda events, objects: ak.pad_none(
        objects["jets_central"][ak.argsort(objects["jets_central"].pt, axis=1, ascending=False)], 4
    ),
    # Padded forward jets (j0forward, j1forward)
    "jets_forward_padded": lambda events, objects: ak.pad_none(
        objects["jets_forward"][ak.argsort(objects["jets_forward"].pt, axis=1, ascending=False)], 2
    ),
    # Combined central + forward jets
    "jets": lambda events, objects: ak.with_name(
        ak.concatenate([objects["jets_central"], objects["jets_forward"]], axis=1),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # B-tagged jets using pre-existing flags from input file
    "bjets_loose": lambda events, objects: objects["jets_central"][objects["jets_central"].isLooseBTag == 1],
    "bjets_medium": lambda events, objects: objects["jets_central"][objects["jets_central"].isMediumBTag == 1],
    "bjets_tight": lambda events, objects: objects["jets_central"][objects["jets_central"].isTightBTag == 1],
    # Central jets ordered by b-tag score
    "jets_central_bscoreordered_padded": lambda events, objects: ak.pad_none(
        objects["jets_central"][ak.argsort(objects["jets_central"].btagDeepFlavB, axis=1, ascending=False)], 2
    ),
    # Jet pairs (central) - computed ONCE
    "jj_pairs_cent": lambda events, objects: ak.combinations(objects["jets_central_padded"], 2, fields=["j0", "j1"]),
    # Jet pairs (forward) - computed ONCE
    "jj_pairs_fwd": lambda events, objects: ak.combinations(objects["jets_forward_padded"], 2, fields=["j0", "j1"]),
}

# Third level derived objects (depend on second level derived objects)
derived_objects_config_level3 = {
    # Padded combined jets
    "jets_padded": lambda events, objects: ak.pad_none(
        objects["jets"][ak.argsort(objects["jets"].pt, axis=1, ascending=False)], 4
    ),
    # Loose b-jets ordered by pt
    "bjets_loose_padded": lambda events, objects: ak.pad_none(
        objects["bjets_loose"][ak.argsort(objects["bjets_loose"].pt, axis=1, ascending=False)], 2
    ),
    # Loose b-jets ordered by b-tag score
    "bjets_loose_bscoreordered_padded": lambda events, objects: ak.pad_none(
        objects["bjets_loose"][ak.argsort(objects["bjets_loose"].btagDeepFlavB, axis=1, ascending=False)], 2
    ),
    # Medium b-jets ordered by b-tag score
    "bjets_medium_bscoreordered_padded": lambda events, objects: ak.pad_none(
        objects["bjets_medium"][ak.argsort(objects["bjets_medium"].btagDeepFlavB, axis=1, ascending=False)], 2
    ),
    # Jet pairs (any) - computed ONCE
    "jj_pairs_any": lambda events, objects: ak.combinations(objects["jets_padded"], 2, fields=["j0", "j1"]),
    # Jet pairs (all jets, not padded) - computed ONCE
    "jj_pairs_all": lambda events, objects: ak.combinations(objects["jets"], 2, fields=["j0", "j1"]),
    # Jet triplets (central) - computed ONCE
    "jjj_triplets_cent": lambda events, objects: ak.combinations(objects["jets_central_padded"], 3, fields=["j0", "j1", "j2"]),
    # Jet triplets (any) - computed ONCE
    "jjj_triplets_any": lambda events, objects: ak.combinations(objects["jets_padded"], 3, fields=["j0", "j1", "j2"]),
}

# Fourth level - pre-computed derived quantities from combinations
derived_objects_config_level4 = {
    # Mjj masses for all pairs
    "jj_pairs_cent_mass": lambda events, objects: (objects["jj_pairs_cent"].j0 + objects["jj_pairs_cent"].j1).mass,
    "jj_pairs_fwd_mass": lambda events, objects: (objects["jj_pairs_fwd"].j0 + objects["jj_pairs_fwd"].j1).mass,
    "jj_pairs_any_mass": lambda events, objects: (objects["jj_pairs_any"].j0 + objects["jj_pairs_any"].j1).mass,
    # Delta R for any jet pairs
    "jj_pairs_any_dr": lambda events, objects: objects["jj_pairs_any"].j0.delta_r(objects["jj_pairs_any"].j1),
    # Delta eta for all jet pairs
    "jj_pairs_all_deta": lambda events, objects: abs(objects["jj_pairs_all"].j0.eta - objects["jj_pairs_all"].j1.eta),
    # Triplet masses
    "jjj_triplets_cent_mass": lambda events, objects: (objects["jjj_triplets_cent"].j0 + objects["jjj_triplets_cent"].j1 + objects["jjj_triplets_cent"].j2).mass,
    "jjj_triplets_any_mass": lambda events, objects: (objects["jjj_triplets_any"].j0 + objects["jjj_triplets_any"].j1 + objects["jjj_triplets_any"].j2).mass,
}

##################################################
## Define dense variables to plot
##################################################
dense_variables_config = { #name of axis must be same as key

    ##################################################
    # MET
    ##################################################
    # "met_significance": {
    #     "axis": axis.Regular(100, 0, 150, name="met_significance", label="MET $\sigma$"),
    #     "expr": lambda events, objects: events.met.significance
    # },
    "met_pt": {
        "axis": axis.Regular(100, 0, 1000, name="met_pt", label="MET $p_T$ [GeV]"),
        "expr": lambda events, objects: events.met.pt
    },
    "met_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="met_phi", label="MET $\phi$"),
        "expr": lambda events, objects: events.met.phi
    },

    ##################################################
    # Jet counts and HT
    ##################################################
    "njets_central": {
        "axis": axis.Regular(25, 0, 25, name="njets_central", label="N AK4 jets (central)"),
        "expr": lambda events, objects: ak.num(objects["jets_central"]),
    },
    "njets_forward": {
        "axis": axis.Regular(15, 0, 15, name="njets_forward", label="N AK4 jets (forward)"),
        "expr": lambda events, objects: ak.num(objects["jets_forward"]),
    },
    "njets_tot": {
        "axis": axis.Regular(25, 0, 25, name="njets_tot", label="N AK4 jets (total)"),
        "expr": lambda events, objects: ak.num(objects["jets"]),
    },
    "nfatjets": {
        "axis": axis.Regular(6, 0, 6, name="nfatjets", label="N AK8 jets"),
        "expr": lambda events, objects: ak.num(events.fatjet)
    },
    "njets_plus_2nfatjets": {
        "axis": axis.Regular(16, 0, 16, name="njets_plus_2nfatjets", label="2*(N AK8 jets) + N AK jets"),
        "expr": lambda events, objects: 2*ak.num(events.fatjet) + ak.num(objects["jets"])
    },
    "scalarptsum_jetCent": {
        "axis": axis.Regular(100, 0, 2000, name="scalarptsum_jetCent", label="$H_T$ (central jets) [GeV]"),
        "expr": lambda events, objects: ak.sum(objects["jets_central"].pt, axis=-1)
    },
    "scalarptsum_jets": {
        "axis": axis.Regular(100, 0, 2000, name="scalarptsum_jets", label="$H_T$ (all jets) [GeV]"),
        "expr": lambda events, objects: ak.sum(objects["jets"].pt, axis=-1)
    },
    "scalarptsum_jetFwd": {
        "axis": axis.Regular(100, 0, 1000, name="scalarptsum_jetFwd", label="$H_T$ (forward jets) [GeV]"),
        "expr": lambda events, objects: ak.sum(objects["jets_forward"].pt, axis=-1)
    },


    ##################################################
    # Central jets j0_central, j1_central, j2_central, j3_central
    ##################################################
    # "j0_central_pt": {
    #     "axis": axis.Regular(100, 0, 500, name="j0_central_pt", label="j0 (central) $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,0].pt
    # },
    # "j0_central_eta": {
    #     "axis": axis.Regular(40, -3, 3, name="j0_central_eta", label="j0 (central) $\eta$"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,0].eta
    # },
    # "j0_central_phi": {
    #     "axis": axis.Regular(40, -3.5, 3.5, name="j0_central_phi", label="j0 (central) $\phi$"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,0].phi
    # },
    # "j1_central_pt": {
    #     "axis": axis.Regular(100, 0, 300, name="j1_central_pt", label="j1 (central) $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,1].pt
    # },
    # "j1_central_eta": {
    #     "axis": axis.Regular(40, -3, 3, name="j1_central_eta", label="j1 (central) $\eta$"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,1].eta
    # },
    # "j2_central_pt": {
    #     "axis": axis.Regular(100, 0, 200, name="j2_central_pt", label="j2 (central) $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,2].pt
    # },
    # "j3_central_pt": {
    #     "axis": axis.Regular(100, 0, 150, name="j3_central_pt", label="j3 (central) $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["jets_central_padded"][:,3].pt
    # },

    ##################################################
    # Forward jets j0_forward, j1_forward
    ##################################################
    # "j0_forward_pt": {
    #     "axis": axis.Regular(100, 0, 300, name="j0_forward_pt", label="j0 (forward) $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["jets_forward_padded"][:,0].pt
    # },
    # "j0_forward_eta": {
    #     "axis": axis.Regular(40, -5, 5, name="j0_forward_eta", label="j0 (forward) $\eta$"),
    #     "expr": lambda events, objects: objects["jets_forward_padded"][:,0].eta
    # },
    # "j0_forward_phi": {
    #     "axis": axis.Regular(40, -3.5, 3.5, name="j0_forward_phi", label="j0 (forward) $\phi$"),
    #     "expr": lambda events, objects: objects["jets_forward_padded"][:,0].phi
    # },
    # "j1_forward_pt": {
    #     "axis": axis.Regular(100, 0, 200, name="j1_forward_pt", label="j1 (forward) $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["jets_forward_padded"][:,1].pt
    # },

    ##################################################
    # Any jets (central + forward) j0, j1, j2, j3
    ##################################################
    "j0_pt": {
        "axis": axis.Regular(100, 0, 500, name="j0_pt", label="j0 $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["jets_padded"][:,0].pt
    },
    # "j0_eta": {
    #     "axis": axis.Regular(40, -5, 5, name="j0_eta", label="j0 $\eta$"),
    #     "expr": lambda events, objects: objects["jets_padded"][:,0].eta
    # },
    "j0_met_dphi":{
        "axis": axis.Regular(50, 0, 3.5, name="j0_met_dphi", label="$\Delta\phi$(j0, MET)"),
        "expr": lambda events, objects: abs(objects["jets_padded"][:,0].delta_phi(events.met))
    },
    "j1_pt": {
        "axis": axis.Regular(100, 0, 300, name="j1_pt", label="j1 $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["jets_padded"][:,1].pt
    },
    "j2_pt": {
        "axis": axis.Regular(100, 0, 200, name="j2_pt", label="j2 $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["jets_padded"][:,2].pt
    },
    "j3_pt": {
        "axis": axis.Regular(100, 0, 150, name="j3_pt", label="j3 $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["jets_padded"][:,3].pt
    },

    ##################################################
    # Fatjets fj0, fj1 (pt-ordered)
    ##################################################
    # "fj0_pt": {
    #     "axis": axis.Regular(100, 0, 2000, name="fj0_pt", label="fj0 $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,0].pt
    # },
    # "fj0_eta": {
    #     "axis": axis.Regular(40, -3, 3, name="fj0_eta", label="fj0 $\eta$"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,0].eta
    # },
    # "fj0_phi": {
    #     "axis": axis.Regular(40, -3.5, 3.5, name="fj0_phi", label="fj0 $\phi$"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,0].phi
    # },
    # "fj0_mass": {
    #     "axis": axis.Regular(100, 0, 300, name="fj0_mass", label="fj0 mass [GeV]"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,0].mass
    # },
    "fj0_met_dphi":{
        "axis": axis.Regular(50, 0, 3.5, name="fj0_met_dphi", label="$\Delta\phi$(fj0, MET)"),
        "expr": lambda events, objects: abs(objects["fatjets_padded"][:,0].delta_phi(events.met))
    },
    "fj0_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="fj0_msoftdrop", label="fj0 softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].msoftdrop
    },
    "fj0_HbbScore": {
        "axis": axis.Regular(50, 0, 1, name="fj0_HbbScore", label="fj0 Hbb Score"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].HbbScore
    },
    "fj0_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="fj0_WqqScore", label="fj0 Wqq Score"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].WqqScore
    },
    "fj0_particleNet_TvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="fj0_particleNet_TvsQCD", label="fj0 particleNet TvsQCD"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].particleNet_TvsQCD
    },
    "fj0_particleNet_WvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="fj0_particleNet_WvsQCD", label="fj0 particleNet WvsQCD"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].particleNet_WvsQCD
    },
    "fj0_particleNet_ZvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="fj0_particleNet_ZvsQCD", label="fj0 particleNet ZvsQCD"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].particleNet_ZvsQCD
    },
    # H vs W/Z discriminant for leading fatjet (for 1-FJ categorization)
    # HvsWZ = Hbb / (Hbb + Wqq + eps), ranges from 0 to 1
    # > 0.5 means more Higgs-like, <= 0.5 means more W/Z-like
    "fj0_HvsWZ_score": {
        "axis": axis.Regular(50, 0, 1, name="fj0_HvsWZ_score", label="fj0 H vs W/Z score"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,0].HbbScore / (
            objects["fatjets_padded"][:,0].HbbScore + objects["fatjets_padded"][:,0].WqqScore + 1e-6
        )
    },
    # "fj1_pt": {
    #     "axis": axis.Regular(100, 0, 1500, name="fj1_pt", label="fj1 $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,1].pt
    # },
    # "fj1_eta": {
    #     "axis": axis.Regular(40, -3, 3, name="fj1_eta", label="fj1 $\eta$"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,1].eta
    # },
    # "fj1_phi": {
    #     "axis": axis.Regular(40, -3.5, 3.5, name="fj1_phi", label="fj1 $\phi$"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,1].phi
    # },
    # "fj1_mass": {
    #     "axis": axis.Regular(100, 0, 300, name="fj1_mass", label="fj1 mass [GeV]"),
    #     "expr": lambda events, objects: objects["fatjets_padded"][:,1].mass
    # },
    "fj1_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="fj1_msoftdrop", label="fj1 softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,1].msoftdrop
    },
    "fj1_HbbScore": {
        "axis": axis.Regular(50, 0, 1, name="fj1_HbbScore", label="fj1 Hbb Score"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,1].HbbScore
    },
    "fj1_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="fj1_WqqScore", label="fj1 Wqq Score"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,1].WqqScore
    },
    "fj1_particleNet_TvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="fj1_particleNet_TvsQCD", label="fj1 particleNet TvsQCD"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,1].particleNet_TvsQCD
    },
    "fj1_particleNet_WvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="fj1_particleNet_WvsQCD", label="fj1 particleNet WvsQCD"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,1].particleNet_WvsQCD
    },
    "fj1_particleNet_ZvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="fj1_particleNet_ZvsQCD", label="fj1 particleNet ZvsQCD"),
        "expr": lambda events, objects: objects["fatjets_padded"][:,1].particleNet_ZvsQCD
    },


    ##################################################
    # Max Hbb Score AK8
    ##################################################
    # "maxHbbAK8_pt": {
    #     "axis": axis.Regular(100, 0, 2000, name="maxHbbAK8_pt", label="Max Hbb AK8 $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["maxHbbAK8"].pt
    # },
    # "maxHbbAK8_eta": {
    #     "axis": axis.Regular(40, -3, 3, name="maxHbbAK8_eta", label="Max Hbb AK8 $\eta$"),
    #     "expr": lambda events, objects: objects["maxHbbAK8"].eta
    # },
    # "maxHbbAK8_phi": {
    #     "axis": axis.Regular(40, -3.5, 3.5, name="maxHbbAK8_phi", label="Max Hbb AK8 $\phi$"),
    #     "expr": lambda events, objects: objects["maxHbbAK8"].phi
    # },
    # "maxHbbAK8_mass": {
    #     "axis": axis.Regular(100, 0, 300, name="maxHbbAK8_mass", label="Max Hbb AK8 mass [GeV]"),
    #     "expr": lambda events, objects: objects["maxHbbAK8"].mass
    # },
    "maxHbbAK8_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="maxHbbAK8_msoftdrop", label="Max Hbb AK8 softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["maxHbbAK8"].msoftdrop
    },
    "maxHbbAK8_HbbScore": {
        "axis": axis.Regular(50, 0, 1, name="maxHbbAK8_HbbScore", label="Max Hbb AK8 Hbb Score"),
        "expr": lambda events, objects: objects["maxHbbAK8"].HbbScore
    },
    "maxHbbAK8_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="maxHbbAK8_WqqScore", label="Max Hbb AK8 Wqq Score"),
        "expr": lambda events, objects: objects["maxHbbAK8"].WqqScore
    },
    "maxHbbAK8_particleNet_TvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="maxHbbAK8_particleNet_TvsQCD", label="Max Hbb AK8 particleNet TvsQCD"),
        "expr": lambda events, objects: objects["maxHbbAK8"].particleNet_TvsQCD
    },
    "maxHbbAK8_particleNet_WvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="maxHbbAK8_particleNet_WvsQCD", label="Max Hbb AK8 particleNet WvsQCD"),
        "expr": lambda events, objects: objects["maxHbbAK8"].particleNet_WvsQCD
    },
    "maxHbbAK8_particleNet_ZvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="maxHbbAK8_particleNet_ZvsQCD", label="Max Hbb AK8 particleNet ZvsQCD"),
        "expr": lambda events, objects: objects["maxHbbAK8"].particleNet_ZvsQCD
    },

    ##################################################
    # Max Wqq Score AK8
    ##################################################
    # "maxWqqAK8_pt": {
    #     "axis": axis.Regular(100, 0, 2000, name="maxWqqAK8_pt", label="Max Wqq AK8 $p_T$ [GeV]"),
    #     "expr": lambda events, objects: objects["maxWqqAK8"].pt
    # },
    # "maxWqqAK8_eta": {
    #     "axis": axis.Regular(40, -3, 3, name="maxWqqAK8_eta", label="Max Wqq AK8 $\eta$"),
    #     "expr": lambda events, objects: objects["maxWqqAK8"].eta
    # },
    # "maxWqqAK8_phi": {
    #     "axis": axis.Regular(40, -3.5, 3.5, name="maxWqqAK8_phi", label="Max Wqq AK8 $\phi$"),
    #     "expr": lambda events, objects: objects["maxWqqAK8"].phi
    # },
    # "maxWqqAK8_mass": {
    #     "axis": axis.Regular(100, 0, 300, name="maxWqqAK8_mass", label="Max Wqq AK8 mass [GeV]"),
    #     "expr": lambda events, objects: objects["maxWqqAK8"].mass
    # },
    "maxWqqAK8_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="maxWqqAK8_msoftdrop", label="Max Wqq AK8 softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["maxWqqAK8"].msoftdrop
    },
    "maxWqqAK8_HbbScore": {
        "axis": axis.Regular(50, 0, 1, name="maxWqqAK8_HbbScore", label="Max Wqq AK8 Hbb Score"),
        "expr": lambda events, objects: objects["maxWqqAK8"].HbbScore
    },
    "maxWqqAK8_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="maxWqqAK8_WqqScore", label="Max Wqq AK8 Wqq Score"),
        "expr": lambda events, objects: objects["maxWqqAK8"].WqqScore
    },
    "maxWqqAK8_particleNet_TvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="maxWqqAK8_particleNet_TvsQCD", label="Max Wqq AK8 particleNet TvsQCD"),
        "expr": lambda events, objects: objects["maxWqqAK8"].particleNet_TvsQCD
    },
    "maxWqqAK8_particleNet_WvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="maxWqqAK8_particleNet_WvsQCD", label="Max Wqq AK8 particleNet WvsQCD"),
        "expr": lambda events, objects: objects["maxWqqAK8"].particleNet_WvsQCD
    },
    "maxWqqAK8_particleNet_ZvsQCD": {
        "axis": axis.Regular(50, 0, 1, name="maxWqqAK8_particleNet_ZvsQCD", label="Max Wqq AK8 particleNet ZvsQCD"),
        "expr": lambda events, objects: objects["maxWqqAK8"].particleNet_ZvsQCD
    },

    ##################################################
    # Multi-jet mass variables (using pre-computed padded jets)
    ##################################################
    # "mjjjcnt": {
    #     "axis": axis.Regular(100, 0, 3000, name="mjjjcnt", label="$m_{jjj}$ (central jets) [GeV]"),
    #     "expr": lambda events, objects: ak.where(
    #         ak.num(objects["jets_central"]) >= 3,
    #         (objects["jets_central_padded"][:,0] + objects["jets_central_padded"][:,1] + objects["jets_central_padded"][:,2]).mass,
    #         0
    #     )
    # },
    # "mjjj": {
    #     "axis": axis.Regular(100, 0, 3000, name="mjjj", label="$m_{jjj}$ (any jets) [GeV]"),
    #     "expr": lambda events, objects: ak.where(
    #         ak.num(objects["jets"]) >= 3,
    #         (objects["jets_padded"][:,0] + objects["jets_padded"][:,1] + objects["jets_padded"][:,2]).mass,
    #         0
    #     )
    # },
    # "mjjjjcnt": {
    #     "axis": axis.Regular(100, 0, 4000, name="mjjjjcnt", label="$m_{jjjj}$ (central jets) [GeV]"),
    #     "expr": lambda events, objects: ak.where(
    #         ak.num(objects["jets_central"]) >= 4,
    #         (objects["jets_central_padded"][:,0] + objects["jets_central_padded"][:,1] + objects["jets_central_padded"][:,2] + objects["jets_central_padded"][:,3]).mass,
    #         0
    #     )
    # },
    # "mjjjj": {
    #     "axis": axis.Regular(100, 0, 4000, name="mjjjj", label="$m_{jjjj}$ (any jets) [GeV]"),
    #     "expr": lambda events, objects: ak.where(
    #         ak.num(objects["jets"]) >= 4,
    #         (objects["jets_padded"][:,0] + objects["jets_padded"][:,1] + objects["jets_padded"][:,2] + objects["jets_padded"][:,3]).mass,
    #         0
    #     )
    # },

    ##################################################
    # B-tagging counts
    ##################################################
    "nbtagsl": {
        "axis": axis.Regular(10, 0, 10, name="nbtagsl", label="N b-jets (loose)"),
        "expr": lambda events, objects: ak.num(objects["bjets_loose"])
    },
    "nbtagsm": {
        "axis": axis.Regular(10, 0, 10, name="nbtagsm", label="N b-jets (medium)"),
        "expr": lambda events, objects: ak.num(objects["bjets_medium"])
    },
    "nbtagst": {
        "axis": axis.Regular(10, 0, 10, name="nbtagst", label="N b-jets (tight)"),
        "expr": lambda events, objects: ak.num(objects["bjets_tight"])
    },

    ##################################################
    # B-jet masses (pt-ordered)
    ##################################################
    "mass_b0b1": {
        "axis": axis.Regular(100, 0, 500, name="mass_b0b1", label="$m_{bb}$ (pt-ordered loose b-jets) [GeV]"),
        "expr": lambda events, objects: ak.where(
            ak.num(objects["bjets_loose"]) > 1,
            (objects["bjets_loose_padded"][:,0] + objects["bjets_loose_padded"][:,1]).mass,
            0
        )
    },

    ##################################################
    # B-jet variables (b-score ordered, loose WP)
    ##################################################
    "bbscore0_bscore": {
        "axis": axis.Regular(50, 0, 1, name="bbscore0_bscore", label="b-score of leading b-jet (loose)"),
        "expr": lambda events, objects: ak.fill_none(objects["bjets_loose_bscoreordered_padded"][:,0].btagDeepFlavB, 0)
    },
    "bbscore1_bscore": {
        "axis": axis.Regular(50, 0, 1, name="bbscore1_bscore", label="b-score of subleading b-jet (loose)"),
        "expr": lambda events, objects: ak.fill_none(objects["bjets_loose_bscoreordered_padded"][:,1].btagDeepFlavB, 0)
    },
    "mass_bbscore0bbscore1": {
        "axis": axis.Regular(100, 0, 500, name="mass_bbscore0bbscore1", label="$m_{bb}$ (b-score ordered loose b-jets) [GeV]"),
        "expr": lambda events, objects: ak.fill_none(
            (objects["bjets_loose_bscoreordered_padded"][:,0] + objects["bjets_loose_bscoreordered_padded"][:,1]).mass,
            0
        )
    },

    ##################################################
    # B-jet variables (b-score ordered, medium WP)
    ##################################################
    "mass_bmbscore0bmbscore1": {
        "axis": axis.Regular(100, 0, 500, name="mass_bmbscore0bmbscore1", label="$m_{bb}$ (b-score ordered medium b-jets) [GeV]"),
        "expr": lambda events, objects: ak.fill_none(
            (objects["bjets_medium_bscoreordered_padded"][:,0] + objects["bjets_medium_bscoreordered_padded"][:,1]).mass,
            0
        )
    },

    ##################################################
    # Central jet b-scores (b-score ordered)
    ##################################################
    "jbscore0_bscore": {
        "axis": axis.Regular(50, 0, 1, name="jbscore0_bscore", label="b-score of central jet with highest b-score"),
        "expr": lambda events, objects: ak.fill_none(objects["jets_central_bscoreordered_padded"][:,0].btagDeepFlavB, 0)
    },
    "jbscore1_bscore": {
        "axis": axis.Regular(50, 0, 1, name="jbscore1_bscore", label="b-score of central jet with 2nd highest b-score"),
        "expr": lambda events, objects: ak.fill_none(objects["jets_central_bscoreordered_padded"][:,1].btagDeepFlavB, 0)
    },
    "mass_jbscore0jbscore1": {
        "axis": axis.Regular(100, 0, 500, name="mass_jbscore0jbscore1", label="$m_{jj}$ (b-score ordered central jets) [GeV]"),
        "expr": lambda events, objects: ak.fill_none(
            (objects["jets_central_bscoreordered_padded"][:,0] + objects["jets_central_bscoreordered_padded"][:,1]).mass,
            0
        )
    },

    ##################################################
    # Mjj max variables (using pre-computed pair masses)
    ##################################################
    # "mjj_max_cent": {
    #     "axis": axis.Regular(100, 0, 1000, name="mjj_max_cent", label="Max $m_{jj}$ (central jets) [GeV]"),
    #     "expr": lambda events, objects: ak.fill_none(ak.max(objects["jj_pairs_cent_mass"], axis=-1), 0)
    # },
    # "mjj_max_fwd": {
    #     "axis": axis.Regular(100, 0, 2500, name="mjj_max_fwd", label="Max $m_{jj}$ (forward jets) [GeV]"),
    #     "expr": lambda events, objects: ak.fill_none(ak.max(objects["jj_pairs_fwd_mass"], axis=-1), 0)
    # },
    "mjj_max": {
        "axis": axis.Regular(100, 0, 2000, name="mjj_max", label="Max $m_{jj}$ (any jets) [GeV]"),
        "expr": lambda events, objects: ak.fill_none(ak.max(objects["jj_pairs_any_mass"], axis=-1), 0)
    },
    "deta_max": {
        "axis": axis.Regular(100, 0, 10, name="deta_max", label="Max $|\Delta\eta_{jj}|$ (all jets)"),
        "expr": lambda events, objects: ak.fill_none(ak.max(objects["jj_pairs_all_deta"], axis=-1), 0)
    },

    ##################################################
    # Min deltaR jet pair mjj (using pre-computed pairs)
    ##################################################
    "jj_pairs_atmindr_mjj": {
        "axis": axis.Regular(100, 0, 1000, name="jj_pairs_atmindr_mjj", label="$m_{jj}$ at min $\Delta R$ [GeV]"),
        "expr": lambda events, objects: ak.flatten(ak.fill_none(
            objects["jj_pairs_any_mass"][ak.argmin(objects["jj_pairs_any_dr"], axis=1, keepdims=True)],
            -999
        ))
    },

    ##################################################
    # Jet triplet masses closest to top (using pre-computed triplet masses)
    ##################################################
    "mjjj_nearest_t": {
        "axis": axis.Regular(100, 0, 700, name="mjjj_nearest_t", label="$m_{jjj}$ closest to top (any jets) [GeV]"),
        "expr": lambda events, objects: ak.fill_none(
            ak.flatten(objects["jjj_triplets_any_mass"][ak.argmin(abs(objects["jjj_triplets_any_mass"] - 173), keepdims=True, axis=1)]),
            0
        )
    },
    "mjjjcnt_nearest_t": {
        "axis": axis.Regular(100, 0, 700, name="mjjjcnt_nearest_t", label="$m_{jjj}$ closest to top (central jets) [GeV]"),
        "expr": lambda events, objects: ak.fill_none(
            ak.flatten(objects["jjj_triplets_cent_mass"][ak.argmin(abs(objects["jjj_triplets_cent_mass"] - 173), keepdims=True, axis=1)]),
            0
        )
    },
}

##################################################
## Helper functions for truth-matched objects
##################################################
def get_object_by_idx(objects, idx):
    """
    Get object (jet or fatjet) from array using truth index.
    Returns None where idx < 0 (no match found) or idx >= number of objects.

    Args:
        objects: Jagged array of jets/fatjets (one list per event)
        idx: 1D array of indices (one per event), -1 means no match

    How it works:
        - ak.singletons() wraps each index in a list: [3,1,2] -> [[3],[1],[2]]
          This is needed because objects[...] expects jagged indices for per-event selection
        - We check both idx >= 0 AND idx < num_objects to avoid out-of-bounds errors
        - Invalid indices use 0 as placeholder, then get masked with ak.mask
        - ak.mask preserves the record type (so .pt, .eta etc. still work) while
          setting invalid entries to None
    """
    n_objects = ak.num(objects)
    valid = (idx >= 0) & (idx < n_objects)
    safe_idx = ak.where(valid, idx, 0)  # Replace invalid indices with 0 temporarily
    selected = objects[ak.singletons(safe_idx)][:, 0]  # Select and flatten
    return ak.mask(selected, valid)  # Mask invalid selections (preserves record type)


##################################################
## Truth-matched objects config (signal only)
## These are built from truth indices matching gen particles to reco jets
## NOTE: NanoAODSchema groups branches by prefix, so truth_h_idx in the ROOT
## file becomes events.truth.h_idx in coffea (not events.truth_h_idx)
##################################################
truth_objects_config = {
    # Fatjets matched to H, V1, V2
    "truth_h_fj": lambda events, objects: ak.with_name(
        get_object_by_idx(events.fatjet, events.truth.h_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "truth_v1_fj": lambda events, objects: ak.with_name(
        get_object_by_idx(events.fatjet, events.truth.v1_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "truth_v2_fj": lambda events, objects: ak.with_name(
        get_object_by_idx(events.fatjet, events.truth.v2_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Jets matched to VBS quarks
    "truth_vbs1_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.vbs1_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "truth_vbs2_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.vbs2_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Jets matched to b quarks from H->bb
    "truth_b1_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.b1_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "truth_b2_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.b2_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Jets matched to V1 decay quarks
    "truth_v1q1_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.v1q1_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "truth_v1q2_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.v1q2_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # Jets matched to V2 decay quarks
    "truth_v2q1_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.v2q1_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "truth_v2q2_j": lambda events, objects: ak.with_name(
        get_object_by_idx(events.jet, events.truth.v2q2_idx),
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
}


##################################################
## Truth-matched histogram variables (signal only)
##################################################
dense_truth_variables_config = {
    ##################################################
    # Truth-matched Higgs fatjet
    ##################################################
    "truth_h_fj_pt": {
        "axis": axis.Regular(100, 0, 2000, name="truth_h_fj_pt", label="Truth H fatjet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_h_fj"].pt
    },
    "truth_h_fj_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_h_fj_eta", label="Truth H fatjet $\\eta$"),
        "expr": lambda events, objects: objects["truth_h_fj"].eta
    },
    "truth_h_fj_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_h_fj_phi", label="Truth H fatjet $\\phi$"),
        "expr": lambda events, objects: objects["truth_h_fj"].phi
    },
    "truth_h_fj_mass": {
        "axis": axis.Regular(100, 0, 300, name="truth_h_fj_mass", label="Truth H fatjet mass [GeV]"),
        "expr": lambda events, objects: objects["truth_h_fj"].mass
    },
    "truth_h_fj_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="truth_h_fj_msoftdrop", label="Truth H fatjet softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["truth_h_fj"].msoftdrop
    },
    "truth_h_fj_HbbScore": {
        "axis": axis.Regular(50, 0, 1, name="truth_h_fj_HbbScore", label="Truth H fatjet Hbb Score"),
        "expr": lambda events, objects: objects["truth_h_fj"].HbbScore
    },
    "truth_h_fj_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="truth_h_fj_WqqScore", label="Truth H fatjet Wqq Score"),
        "expr": lambda events, objects: objects["truth_h_fj"].WqqScore
    },

    ##################################################
    # Truth-matched V1 fatjet
    ##################################################
    "truth_v1_fj_pt": {
        "axis": axis.Regular(100, 0, 2000, name="truth_v1_fj_pt", label="Truth V1 fatjet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_v1_fj"].pt
    },
    "truth_v1_fj_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_v1_fj_eta", label="Truth V1 fatjet $\\eta$"),
        "expr": lambda events, objects: objects["truth_v1_fj"].eta
    },
    "truth_v1_fj_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_v1_fj_phi", label="Truth V1 fatjet $\\phi$"),
        "expr": lambda events, objects: objects["truth_v1_fj"].phi
    },
    "truth_v1_fj_mass": {
        "axis": axis.Regular(100, 0, 300, name="truth_v1_fj_mass", label="Truth V1 fatjet mass [GeV]"),
        "expr": lambda events, objects: objects["truth_v1_fj"].mass
    },
    "truth_v1_fj_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="truth_v1_fj_msoftdrop", label="Truth V1 fatjet softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["truth_v1_fj"].msoftdrop
    },
    "truth_v1_fj_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="truth_v1_fj_WqqScore", label="Truth V1 fatjet Wqq Score"),
        "expr": lambda events, objects: objects["truth_v1_fj"].WqqScore
    },

    ##################################################
    # Truth-matched V2 fatjet
    ##################################################
    "truth_v2_fj_pt": {
        "axis": axis.Regular(100, 0, 2000, name="truth_v2_fj_pt", label="Truth V2 fatjet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_v2_fj"].pt
    },
    "truth_v2_fj_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_v2_fj_eta", label="Truth V2 fatjet $\\eta$"),
        "expr": lambda events, objects: objects["truth_v2_fj"].eta
    },
    "truth_v2_fj_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_v2_fj_phi", label="Truth V2 fatjet $\\phi$"),
        "expr": lambda events, objects: objects["truth_v2_fj"].phi
    },
    "truth_v2_fj_mass": {
        "axis": axis.Regular(100, 0, 300, name="truth_v2_fj_mass", label="Truth V2 fatjet mass [GeV]"),
        "expr": lambda events, objects: objects["truth_v2_fj"].mass
    },
    "truth_v2_fj_msoftdrop": {
        "axis": axis.Regular(100, 0, 300, name="truth_v2_fj_msoftdrop", label="Truth V2 fatjet softdrop mass [GeV]"),
        "expr": lambda events, objects: objects["truth_v2_fj"].msoftdrop
    },
    "truth_v2_fj_WqqScore": {
        "axis": axis.Regular(50, 0, 1, name="truth_v2_fj_WqqScore", label="Truth V2 fatjet Wqq Score"),
        "expr": lambda events, objects: objects["truth_v2_fj"].WqqScore
    },

    ##################################################
    # Truth-matched VBS jets
    ##################################################
    "truth_vbs1_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_vbs1_j_pt", label="Truth VBS1 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_vbs1_j"].pt
    },
    "truth_vbs1_j_eta": {
        "axis": axis.Regular(50, -5, 5, name="truth_vbs1_j_eta", label="Truth VBS1 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_vbs1_j"].eta
    },
    "truth_vbs1_j_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_vbs1_j_phi", label="Truth VBS1 jet $\\phi$"),
        "expr": lambda events, objects: objects["truth_vbs1_j"].phi
    },
    "truth_vbs1_j_mass": {
        "axis": axis.Regular(50, 0, 100, name="truth_vbs1_j_mass", label="Truth VBS1 jet mass [GeV]"),
        "expr": lambda events, objects: objects["truth_vbs1_j"].mass
    },
    "truth_vbs2_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_vbs2_j_pt", label="Truth VBS2 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_vbs2_j"].pt
    },
    "truth_vbs2_j_eta": {
        "axis": axis.Regular(50, -5, 5, name="truth_vbs2_j_eta", label="Truth VBS2 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_vbs2_j"].eta
    },
    "truth_vbs2_j_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_vbs2_j_phi", label="Truth VBS2 jet $\\phi$"),
        "expr": lambda events, objects: objects["truth_vbs2_j"].phi
    },
    "truth_vbs2_j_mass": {
        "axis": axis.Regular(50, 0, 100, name="truth_vbs2_j_mass", label="Truth VBS2 jet mass [GeV]"),
        "expr": lambda events, objects: objects["truth_vbs2_j"].mass
    },
    # VBS dijet variables
    "truth_vbs_mjj": {
        "axis": axis.Regular(100, 0, 3000, name="truth_vbs_mjj", label="Truth VBS $m_{jj}$ [GeV]"),
        "expr": lambda events, objects: (objects["truth_vbs1_j"] + objects["truth_vbs2_j"]).mass
    },
    "truth_vbs_deta": {
        "axis": axis.Regular(50, 0, 10, name="truth_vbs_deta", label="Truth VBS $|\\Delta\\eta_{jj}|$"),
        "expr": lambda events, objects: abs(objects["truth_vbs1_j"].eta - objects["truth_vbs2_j"].eta)
    },
    "truth_vbs_dphi": {
        "axis": axis.Regular(40, 0, 3.5, name="truth_vbs_dphi", label="Truth VBS $|\\Delta\\phi_{jj}|$"),
        "expr": lambda events, objects: abs(objects["truth_vbs1_j"].delta_phi(objects["truth_vbs2_j"]))
    },

    ##################################################
    # Truth-matched b-jets from H->bb
    ##################################################
    "truth_b1_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_b1_j_pt", label="Truth b1 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_b1_j"].pt
    },
    "truth_b1_j_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_b1_j_eta", label="Truth b1 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_b1_j"].eta
    },
    "truth_b1_j_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_b1_j_phi", label="Truth b1 jet $\\phi$"),
        "expr": lambda events, objects: objects["truth_b1_j"].phi
    },
    "truth_b1_j_btagDeepFlavB": {
        "axis": axis.Regular(50, 0, 1, name="truth_b1_j_btagDeepFlavB", label="Truth b1 jet b-tag score"),
        "expr": lambda events, objects: objects["truth_b1_j"].btagDeepFlavB
    },
    "truth_b2_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_b2_j_pt", label="Truth b2 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_b2_j"].pt
    },
    "truth_b2_j_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_b2_j_eta", label="Truth b2 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_b2_j"].eta
    },
    "truth_b2_j_phi": {
        "axis": axis.Regular(40, -3.5, 3.5, name="truth_b2_j_phi", label="Truth b2 jet $\\phi$"),
        "expr": lambda events, objects: objects["truth_b2_j"].phi
    },
    "truth_b2_j_btagDeepFlavB": {
        "axis": axis.Regular(50, 0, 1, name="truth_b2_j_btagDeepFlavB", label="Truth b2 jet b-tag score"),
        "expr": lambda events, objects: objects["truth_b2_j"].btagDeepFlavB
    },
    # bb dijet variables
    "truth_bb_mjj": {
        "axis": axis.Regular(100, 0, 300, name="truth_bb_mjj", label="Truth $m_{bb}$ [GeV]"),
        "expr": lambda events, objects: (objects["truth_b1_j"] + objects["truth_b2_j"]).mass
    },
    "truth_bb_dr": {
        "axis": axis.Regular(50, 0, 5, name="truth_bb_dr", label="Truth $\\Delta R_{bb}$"),
        "expr": lambda events, objects: objects["truth_b1_j"].delta_r(objects["truth_b2_j"])
    },

    ##################################################
    # Truth-matched V1 decay quarks
    ##################################################
    "truth_v1q1_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_v1q1_j_pt", label="Truth V1q1 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_v1q1_j"].pt
    },
    "truth_v1q1_j_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_v1q1_j_eta", label="Truth V1q1 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_v1q1_j"].eta
    },
    "truth_v1q2_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_v1q2_j_pt", label="Truth V1q2 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_v1q2_j"].pt
    },
    "truth_v1q2_j_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_v1q2_j_eta", label="Truth V1q2 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_v1q2_j"].eta
    },
    # V1 qq dijet variables
    "truth_v1qq_mjj": {
        "axis": axis.Regular(100, 0, 200, name="truth_v1qq_mjj", label="Truth V1 $m_{qq}$ [GeV]"),
        "expr": lambda events, objects: (objects["truth_v1q1_j"] + objects["truth_v1q2_j"]).mass
    },
    "truth_v1qq_dr": {
        "axis": axis.Regular(50, 0, 5, name="truth_v1qq_dr", label="Truth V1 $\\Delta R_{qq}$"),
        "expr": lambda events, objects: objects["truth_v1q1_j"].delta_r(objects["truth_v1q2_j"])
    },

    ##################################################
    # Truth-matched V2 decay quarks
    ##################################################
    "truth_v2q1_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_v2q1_j_pt", label="Truth V2q1 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_v2q1_j"].pt
    },
    "truth_v2q1_j_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_v2q1_j_eta", label="Truth V2q1 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_v2q1_j"].eta
    },
    "truth_v2q2_j_pt": {
        "axis": axis.Regular(100, 0, 500, name="truth_v2q2_j_pt", label="Truth V2q2 jet $p_T$ [GeV]"),
        "expr": lambda events, objects: objects["truth_v2q2_j"].pt
    },
    "truth_v2q2_j_eta": {
        "axis": axis.Regular(40, -3, 3, name="truth_v2q2_j_eta", label="Truth V2q2 jet $\\eta$"),
        "expr": lambda events, objects: objects["truth_v2q2_j"].eta
    },
    # V2 qq dijet variables
    "truth_v2qq_mjj": {
        "axis": axis.Regular(100, 0, 200, name="truth_v2qq_mjj", label="Truth V2 $m_{qq}$ [GeV]"),
        "expr": lambda events, objects: (objects["truth_v2q1_j"] + objects["truth_v2q2_j"]).mass
    },
    "truth_v2qq_dr": {
        "axis": axis.Regular(50, 0, 5, name="truth_v2qq_dr", label="Truth V2 $\\Delta R_{qq}$"),
        "expr": lambda events, objects: objects["truth_v2q1_j"].delta_r(objects["truth_v2q2_j"])
    },

    ##################################################
    # H vs W/Z tagger efficiency (for 1-FJ events)
    # Check if score-based selection matches truth assignment
    # h_idx == 0 means the truth Higgs is matched to fj0
    # v1_idx == 0 or v2_idx == 0 means a truth V is matched to fj0
    ##################################################
    # HvsWZ score for fj0, only filled when fj0 IS the truth Higgs
    "fj0_HvsWZ_score_truthH": {
        "axis": axis.Regular(50, 0, 1, name="fj0_HvsWZ_score_truthH", label="fj0 H vs W/Z score (fj0 is truth H)"),
        "expr": lambda events, objects: ak.mask(
            objects["fatjets_padded"][:,0].HbbScore / (
                objects["fatjets_padded"][:,0].HbbScore + objects["fatjets_padded"][:,0].WqqScore + 1e-6
            ),
            events.truth.h_idx == 0
        )
    },
    # HvsWZ score for fj0, only filled when fj0 IS a truth V (V1 or V2)
    "fj0_HvsWZ_score_truthV": {
        "axis": axis.Regular(50, 0, 1, name="fj0_HvsWZ_score_truthV", label="fj0 H vs W/Z score (fj0 is truth V)"),
        "expr": lambda events, objects: ak.mask(
            objects["fatjets_padded"][:,0].HbbScore / (
                objects["fatjets_padded"][:,0].HbbScore + objects["fatjets_padded"][:,0].WqqScore + 1e-6
            ),
            (events.truth.v1_idx == 0) | (events.truth.v2_idx == 0)
        )
    },
}