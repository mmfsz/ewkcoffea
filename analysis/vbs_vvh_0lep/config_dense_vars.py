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


objects_config = { 
    "leadAK8": lambda events: ak.with_name(
        get_leading_jet(events.fatjet), 
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
        ),
    "leadAK4": lambda events: ak.with_name(
        get_leading_jet(events.jet), 
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
        )
}

derived_objects_config = { 
    "deltaPhi": lambda events, objects_config: ak.with_name(
        objects_config["leadAK8"].delta_phi(events.met), 
        name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )
}

##################################################
## Define dense variables to plot
##################################################
dense_variables_config = { #name of axis must be same as key    
    "nGoodAK4": {
        "axis": axis.Regular(25, 0, 25, name="nGoodAK4", label="nGoodAK4"),
        "expr": lambda events, objects: ak.num(events.jet),
    },
    "nGoodAK8": {
        "axis": axis.Regular(6, 0, 6, name="nGoodAK8", label="nGoodAK8"),
        "expr": lambda events, objects: ak.num(events.fatjet)
    },
    "leadAK8_MET_dphi":{
        "axis": axis.Regular(50, 0, 3.5, name="leadAK8_MET_dphi", label="dphi(leadAK8, MET)"),
        "expr": lambda events, objects: objects["leadAK8"].delta_phi(events.met)
    },
    
    "leadAK4_MET_dphi":{ 
        "axis": axis.Regular(50, 0, 3.5, name="leadAK4_MET_dphi", label="dphi(leadAK4, MET)"),
        "expr": lambda events, objects: objects["leadAK4"].delta_phi(events.met)
    },
}