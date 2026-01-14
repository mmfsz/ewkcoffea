import argparse
import pickle
import gzip
import json
import os
import shutil
import fnmatch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import copy

import ewkcoffea.modules.plotting_tools as plt_tools

warnings.filterwarnings("ignore", message="List indexing selection is experimental")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

hist_to_use = "njets_tot" 

HTML_PC = "/home/users/kmohrman/ref_scripts/html_stuff/index.php"

CLR_LST = ['#d55e00', '#e69f00', '#f0e442', '#009e73', '#0072b2', '#56b4e9', '#cc79a7', '#6e3600', '#a17500', '#a39b2f', '#00664f', '#005d87', '#999999', '#8c5d77']

SIGNAL_LST = ["Signal_C2V_1p5_C3_1p0", "Signal_C2V_1p0_C3_1p0", "Signal_C2V_1p0_C3_10p0"]

# Display labels for signals (using LaTeX formatting)
SIGNAL_DISPLAY_NAMES = {
    "Signal_C2V_1p5_C3_1p0": r"$\kappa_{2V}=1.5, \kappa_{\lambda}=1.0$",
    "Signal_C2V_1p0_C3_1p0": r"$\kappa_{2V}=1.0, \kappa_{\lambda}=1.0$",
    "Signal_C2V_1p0_C3_10p0": r"$\kappa_{2V}=1.0, \kappa_{\lambda}=10$",
}

def get_signal_display_name(sig_name):
    """Get the display name for a signal, with fallback to original name."""
    return SIGNAL_DISPLAY_NAMES.get(sig_name, sig_name)

GRP_DICT_FULL = {

    "Signal_C2V_1p5_C3_1p0" : [
        "VBSWWH_OS_C2V_1p5_C3_1p0_13TeV_4f_LO",
        "VBSWWH_SS_C2V_1p5_C3_1p0_13TeV_4f_LO",
        "VBSWZH_C2V_1p5_C3_1p0_13TeV_4f_LO",
        "VBSZZH_C2V_1p5_C3_1p0_13TeV_4f_LO",
    ],
    "Signal_C2V_1p0_C3_1p0" : [
        "VBSWWH_OS_C2V_1p0_C3_1p0_13TeV_4f_LO",
        "VBSWWH_SS_C2V_1p0_C3_1p0_13TeV_4f_LO",
        "VBSWZH_C2V_1p0_C3_1p0_13TeV_4f_LO",
        "VBSZZH_C2V_1p0_C3_1p0_13TeV_4f_LO",
    ],
    "Signal_C2V_1p0_C3_10p0" : [
        "VBSWWH_OS_C2V_1p0_C3_10p0_13TeV_4f_LO",
        "VBSWWH_SS_C2V_1p0_C3_10p0_13TeV_4f_LO",
        "VBSWZH_C2V_1p0_C3_10p0_13TeV_4f_LO",
        "VBSZZH_C2V_1p0_C3_10p0_13TeV_4f_LO",
    ],

    "QCD" : [
        "QCD_HT1000to1500",
        "QCD_HT100to200",
        "QCD_HT1500to2000",
        "QCD_HT2000toInf",
        "QCD_HT200to300",
        "QCD_HT300to500",
        "QCD_HT500to700",
        "QCD_HT50to100", # Has a spike
        "QCD_HT700to1000",
    ],

    "ttbar-hadronic" : [
        "TTToHadronic",
    ],
    "ttbar-semilep" : [
        "TTToSemiLeptonic",
    ],

    "single-t" : [
        "ST_t-channel_antitop_4f",
        "ST_t-channel_top_4f",
        "ST_tW_antitop_5f",
        "ST_tW_top_5f",
    ],

    "ttX" : [
        "ttHTobb_M125",
        "ttHToNonbb_M125",

        "TTWJetsToQQ",
        "TTWW",
        "TTWZ",
    ],

    "W+jets" : [
        "WJetsToQQ_HT-200to400",
        "WJetsToQQ_HT-400to600",
        "WJetsToQQ_HT-600to800",
        "WJetsToQQ_HT-800toInf",

        "EWKWminus2Jets_WToQQ_dipoleRecoilOn",
        "EWKWplus2Jets_WToQQ_dipoleRecoilOn"
    ],
    "Z+jets" : [
        "ZJetsToQQ_HT-200to400",
        "ZJetsToQQ_HT-400to600",
        "ZJetsToQQ_HT-600to800",
        "ZJetsToQQ_HT-800toInf",

        "EWKZ2Jets_ZToLL_M-50",
        "EWKZ2Jets_ZToNuNu_M-50",
        "EWKZ2Jets_ZToQQ_dipoleRecoilOn",
    ],
    "VV" : [
        "WWTo1L1Nu2Q",
        "WWTo4Q",
        "WZJJ_EWK_InclusivePolarization",
        "WZTo1L1Nu2Q",
        "WZTo2Q2L",

        "ZZTo2Nu2Q",
        "ZZTo2Q2L",
        "ZZTo4Q",
    ],

    "VH" : [
        "ZH_HToBB_ZToQQ_M-125",
        "WminusH_HToBB_WToLNu_M-125",
        "WplusH_HToBB_WToLNu_M-125",
        "VHToNonbb_M125",
    ],

    "VVV" : [
        "WWW_4F",
        "WWZ_4F",
        "WZZ",
        "ZZZ",
        
    ],
}


# not used right now
CAT_LST = [
    "all_events",
    #"filter",
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
]


########################
### Helper functions ###

# Append the years to sample names dict
def append_years(sample_dict_base,year_lst):
    out_dict = {}
    for proc_group in sample_dict_base.keys():
        out_dict[proc_group] = []
        for proc_base_name in sample_dict_base[proc_group]:
            for year_str in year_lst:
                out_dict[proc_group].append(f"{year_str}_{proc_base_name}")
    return out_dict


# Get sig and bkg yield in all categories
def get_yields_per_cat(histo_dict,var_name,year_name_lst_to_prepend, cat_lst, use_counts=False):
    print("-->get_yields_per_cat()")
    print(f"var_name = {var_name}")
    out_dict = {}

    # Get the initial grouping dict
    grouping_dict = append_years(GRP_DICT_FULL,year_name_lst_to_prepend)

    # Get list of all of the backgrounds together
    bkg_lst = []
    for grp in grouping_dict:
        if "Signal" not in grp:
            bkg_lst = bkg_lst + grouping_dict[grp]

    # Make the dictionary to get yields for, it includes what's in grouping_dict, plus the backgrounds grouped as one
    groups_to_get_yields_for_dict = copy.deepcopy(grouping_dict)
    print(f"groups_to_get_yields_for_dict = {groups_to_get_yields_for_dict}")
    groups_to_get_yields_for_dict["Background"] = bkg_lst

    # Loop over cats and fill dict of sig and bkg
    for cat in cat_lst:
        out_dict[cat] = {}
        if use_counts: 
            histo_base = histo_dict[var_name][{"systematic":"count", "category":cat}]
        else:
            histo_base = histo_dict[var_name][{"systematic":"nominal", "category":cat}]

        # Get values per proc
        for group_name,group_lst in groups_to_get_yields_for_dict.items():
            print(f"{group_name} : {group_lst}")
            histo = plt_tools.group(histo_base,"process","process",{group_name:group_lst})
            yld = sum(sum(histo.values(flow=True)))
            print(f"yld = {yld}")
            var = sum(sum(histo.variances(flow=True)))
            out_dict[cat][group_name] = [yld,(var)**0.5]

        # Get the metric
        bkg = out_dict[cat]["Background"][0]
        for signal in SIGNAL_LST:
            sig = out_dict[cat][signal][0]
            metric = sig/(bkg)**0.5
            out_dict[cat]["metric_"+signal] = [metric,None] # Don't bother propagating error

    return out_dict


# Make the figures for the vvh study - version 2 for multiple signals
def make_vvh_fig(histo_mc,histo_mc_sig_dict,histo_mc_bkg,title="test",axisrangex=None,log_y=False,fig=None,axes=None):

    # Create the figure only if not provided (for reuse optimization)
    if fig is None or axes is None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(9,10),
            gridspec_kw={"height_ratios": (3, 1, 1, 1)},
            sharex=True
        )
        fig.subplots_adjust(hspace=.07, right=0.65)
    else:
        ax1, ax2, ax3, ax4 = axes
        # Clear all axes for reuse
        for ax in axes:
            ax.clear()
        # Clear any figure-level text
        for txt in fig.texts[:]:
            txt.remove()

    # Plot the stack plot
    # Make sure we have enough colors for all categories
    n_categories = len(histo_mc.axes["process_grp"])
    colors_to_use = CLR_LST[:n_categories] if n_categories <= len(CLR_LST) else CLR_LST

    histo_mc.plot1d(
        stack=True,
        histtype="fill",
        color=colors_to_use,
        ax=ax1,
        zorder=10,
    )

    # Get the errs on MC and plot them by hand on the stack plot
    histo_mc_sum = histo_mc[{"process_grp":sum}]
    mc_arr = histo_mc_sum.values()
    mc_err_arr = np.sqrt(histo_mc_sum.variances())
    err_p = np.append(mc_arr + mc_err_arr, 0)
    err_m = np.append(mc_arr - mc_err_arr, 0)
    bin_edges_arr = histo_mc_sum.axes[0].edges
    bin_centers_arr = histo_mc_sum.axes[0].centers
    ax1.fill_between(bin_edges_arr,err_m,err_p, step='post', facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.0, label='MC stat', hatch='/////', zorder=11)


    ## Draw the normalized shapes and overlay signals ##

    # Get background yield and normalized histogram
    yld_bkg = sum(sum(histo_mc_bkg.values(flow=True)))
    histo_mc_bkg_norm = plt_tools.scale(copy.deepcopy(histo_mc_bkg), "process_grp", {"Background":1.0/yld_bkg})
    histo_mc_bkg_norm.plot1d(color="gray", ax=ax2, zorder=100, label="Background")

    # Define colors for different signals
    sig_colors = ["red", "blue", "green"]

    # Track max shapes for y-axis scaling
    all_sig_norm_values = []

    # Loop over each signal and plot (only plot signals that are present)
    for idx, sig_group_name in enumerate(SIGNAL_LST):
        if sig_group_name not in histo_mc_sig_dict:
            continue
        histo_mc_sig = histo_mc_sig_dict[sig_group_name]
        sig_color = sig_colors[idx % len(sig_colors)]

        # Get signal yield and metric
        yld_sig = sum(sum(histo_mc_sig.values(flow=True)))
        metric = yld_sig/(yld_bkg**0.5)

        # Scale and plot signal
        histo_mc_sig_scale_to_bkg = plt_tools.scale(copy.deepcopy(histo_mc_sig), "process_grp", {sig_group_name:yld_bkg/yld_sig})
        histo_mc_sig_norm = plt_tools.scale(copy.deepcopy(histo_mc_sig), "process_grp", {sig_group_name:1.0/yld_sig})

        sig_display_name = get_signal_display_name(sig_group_name)
        histo_mc_sig_scale_to_bkg.plot1d(color=[sig_color], ax=ax1, zorder=100+idx, label=sig_display_name)
        histo_mc_sig_norm.plot1d(color=sig_color, ax=ax2, zorder=100+idx, label=sig_display_name)

        all_sig_norm_values.extend(sum(histo_mc_sig_norm.values(flow=True)))


    ## Draw the significance for each signal ##
    yld_bkg_arr = sum(histo_mc_bkg.values())

    # Pre-compute cumulative background for significance calculations
    yld_bkg_arr_cum = np.cumsum(yld_bkg_arr)
    yld_bkg_arr_cum_ud = np.cumsum(np.flipud(yld_bkg_arr))
    yld_bkg_arr_cum_ud_flippedback = np.flipud(yld_bkg_arr_cum_ud)
    #yld_bkg_arr_cum_ud = np.flipud(np.cumsum(np.flipud(yld_bkg_arr)))

    # Track max significance for text positioning
    text_y_offset = 0.35
    max_significance_overall = 0

    for idx, sig_group_name in enumerate(SIGNAL_LST):
        if sig_group_name not in histo_mc_sig_dict:
            continue
        histo_mc_sig = histo_mc_sig_dict[sig_group_name]
        sig_color = sig_colors[idx % len(sig_colors)]

        # Get the sig arrays
        yld_sig = sum(sum(histo_mc_sig.values(flow=True)))
        yld_sig_arr = sum(histo_mc_sig.values())

        # Get the cumulative significance, starting from left
        yld_sig_arr_cum = np.cumsum(yld_sig_arr)
        metric_cum = yld_sig_arr_cum/np.sqrt(yld_bkg_arr_cum)
        metric_cum = np.nan_to_num(metric_cum,nan=0,posinf=0)

        # Get the cumulative significance, starting from right
        yld_sig_arr_cum_ud = np.cumsum(np.flipud(yld_sig_arr))
        metric_cum_ud = np.flipud(yld_sig_arr_cum_ud/np.sqrt(yld_bkg_arr_cum_ud))
        metric_cum_ud = np.nan_to_num(metric_cum_ud,nan=0,posinf=0)
        yld_sig_arr_cum_ud = np.flipud(yld_sig_arr_cum_ud) # Flip back so the order is as expected for later use

        # Draw it on the third plot with different markers for each signal
        #ax3.scatter(bin_centers_arr,metric_cum, facecolor='none',edgecolor=sig_color,marker=">", alpha=0.7, zorder=100+idx)
        #ax3.scatter(bin_centers_arr,metric_cum_ud,facecolor='none',edgecolor=sig_color,marker="<", alpha=0.7, zorder=100+idx)
        ax3.plot(bin_centers_arr, metric_cum, 
                color=sig_color, linestyle='-', linewidth=1.5, alpha=0.8, zorder=100+idx, label=f"{sig_display_name} (keep ≤ x)") # "from left"

        ax3.plot(bin_centers_arr, metric_cum_ud, 
                color=sig_color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=100+idx, label=f"{sig_display_name} (keep ≥ x)") # "from right"
        # Track max significance
        max_significance_overall = max(max_significance_overall, max(metric_cum), max(metric_cum_ud))

        # Write the max values on the plot
        max_metric_from_left_idx = np.argmax(metric_cum)
        max_metric_from_right_idx = np.argmax(metric_cum_ud)
        left_max_y = metric_cum[max_metric_from_left_idx]
        right_max_y = metric_cum_ud[max_metric_from_right_idx]
        left_max_x = bin_centers_arr[max_metric_from_left_idx]
        right_max_x = bin_centers_arr[max_metric_from_right_idx]
        left_s_at_max = yld_sig_arr_cum[max_metric_from_left_idx]
        right_s_at_max = yld_sig_arr_cum_ud[max_metric_from_right_idx]
        left_b_at_max = yld_bkg_arr_cum[max_metric_from_left_idx]
        right_b_at_max = yld_bkg_arr_cum_ud_flippedback[max_metric_from_right_idx]

        sig_display_name = get_signal_display_name(sig_group_name)
        # Only write the max significance value for the highest between left and right
        if left_max_y >= right_max_y:
            plt.text(0.15, text_y_offset, f"{sig_display_name} - Max (≤ x): {np.round(left_max_y,3)} (at x={np.round(left_max_x,2)}, sig: {np.round(left_s_at_max,2)}, bkg: {np.round(left_b_at_max,1)})", fontsize=7, transform=fig.transFigure, color=sig_color)
        else:
            plt.text(0.15, text_y_offset, f"{sig_display_name} - Max (≥ x): {np.round(right_max_y,3)} (at x={np.round(right_max_x,2)}, sig: {np.round(right_s_at_max,2)}, bkg: {np.round(right_b_at_max,1)})", fontsize=7, transform=fig.transFigure, color=sig_color)
        text_y_offset -= 0.02


        ## Draw on the fraction of signal retained for each signal ##
        yld_sig_arr_cum_frac = np.cumsum(yld_sig_arr)/yld_sig
        yld_sig_arr_cum_frac_ud = np.flipud(np.cumsum(np.flipud(yld_sig_arr)))/yld_sig
        ax4.plot(bin_centers_arr,yld_sig_arr_cum_frac, color=sig_color, linestyle='-', linewidth=1.5, alpha=0.8, zorder=100+idx)
        ax4.plot(bin_centers_arr,yld_sig_arr_cum_frac_ud, color=sig_color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=100+idx)

        #ax4.scatter(bin_centers_arr,yld_sig_arr_cum_frac_ud,facecolor='none',edgecolor=sig_color,marker="", alpha=0.7, zorder=100+idx)
        #ax4.scatter(bin_centers_arr,yld_sig_arr_cum_frac, facecolor='none',edgecolor=sig_color,marker=">", alpha=0.7, zorder=100+idx)

    ## Legend, scale the axis, set labels, etc ##

    extr = ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    extr = ax2.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="9", frameon=False)
    # Add custom legend for ax3 and ax4 (only for signals that are present)
    from matplotlib.lines import Line2D
    #legend_elements_sig = [Line2D([0], [0], marker='>', color='w', markerfacecolor='none', markeredgecolor=sig_colors[i % len(sig_colors)], markersize=8, label=get_signal_display_name(sig_name)) for i, sig_name in enumerate(SIGNAL_LST) if sig_name in histo_mc_sig_dict]
    #legend_elements_dir = [Line2D([0], [0], marker='>', color='w', markerfacecolor='none',markeredgecolor='black', markersize=8, label='Cum. from left'),
                          #Line2D([0], [0], marker='<', color='w', markerfacecolor='none',markeredgecolor='black', markersize=8, label='Cum. from right')]
    #extr = ax3.legend(handles=legend_elements_dir, loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    #extr = ax4.legend(handles=legend_elements_dir, loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    #extr = ax3.legend(handles=legend_elements_sig + legend_elements_dir, loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    #extr = ax4.legend(handles=legend_elements_sig + legend_elements_dir, loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)

    # use same legend for ax3 and ax4
    extr = ax3.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    
    # Print yield info for all signals
    text_y_pos = 0.85
    for idx, sig_group_name in enumerate(SIGNAL_LST):
        if sig_group_name not in histo_mc_sig_dict:
            continue
        yld_sig = sum(sum(histo_mc_sig_dict[sig_group_name].values(flow=True)))
        metric = yld_sig/(yld_bkg**0.5)
        sig_display_name = get_signal_display_name(sig_group_name)
        plt.text(0.15, text_y_pos, f"{sig_display_name} yield: {np.round(yld_sig,2)}, metric: {np.round(metric,3)}", fontsize=10, transform=fig.transFigure, color=sig_colors[idx % len(sig_colors)])
        text_y_pos -= 0.02

    plt.text(0.15, text_y_pos, f"Bkg. yield: {np.round(yld_bkg,2)}", fontsize=10, transform=fig.transFigure)
    text_y_pos -= 0.02

    # Calculate average scale factor for note (only for signals that are present)
    if histo_mc_sig_dict:
        avg_scale = np.mean([yld_bkg/sum(sum(histo_mc_sig_dict[sig].values(flow=True))) for sig in SIGNAL_LST if sig in histo_mc_sig_dict])
        plt.text(0.15, text_y_pos, f"[Note: sig. overlays scaled ~{np.round(avg_scale,1)}x on avg]", fontsize=10, transform=fig.transFigure)

    extt = ax1.set_title(title)
    ax1.set_xlabel(None)
    ax2.set_xlabel(None)
    extb = ax3.set_xlabel(None)
    # Plot a dummy hist on ax4 to get the label to show up
    histo_mc.plot1d(alpha=0, ax=ax4)

    extl = ax2.set_ylabel('Shapes')
    ax3.set_ylabel('Significance')
    ax4.set_ylabel('Signal kept (%)')
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax3.axhline(0.0,linestyle="-",color="k",linewidth=0.5)
    ax4.axhline(0.0,linestyle="-",color="k",linewidth=0.5)
    #ax1.grid() # Note: grid does not respect z order :(
    #ax2.grid()
    ax3.grid()
    ax4.grid()

    shapes_ymax = max(max(sum(histo_mc_bkg_norm.values(flow=True))), max(all_sig_norm_values) if all_sig_norm_values else 0)
    significance_max = max_significance_overall
    significance_min = 0-0.1*significance_max
    ax1.autoscale(axis='y')
    ax2.set_ylim(0.0,1.5*shapes_ymax)
    ax3.set_ylim(significance_min,2.5*significance_max)
    ax4.set_ylim(-0.1,1.2)
    if log_y:
        ax1.set_yscale('log')

    if axisrangex is not None:
        ax1.set_xlim(axisrangex[0],axisrangex[1])
        ax2.set_xlim(axisrangex[0],axisrangex[1])


    return (fig, (ax1, ax2, ax3, ax4), (extt,extr,extb,extl))


##############################################################
### Wrapper functions for each of the main functionalities ###


### Sanity check of the different reweight points (for a hist that has extra axis to store that) ###
# Old
def check_rwgt(histo_dict):

    #pkl_file_path = "/home/users/kmohrman/vbs_vvh/ewkcoffea_for_vbs_vvh/ewkcoffea/analysis/vbs_vvh/histos/check_wgt_genw.pkl.gz"
    #pkl_file_path = "/home/users/kmohrman/vbs_vvh/ewkcoffea_for_vbs_vvh/ewkcoffea/analysis/vbs_vvh/histos/check_wgt_sm.pkl.gz"
    #pkl_file_path = "/home/users/kmohrman/vbs_vvh/ewkcoffea_for_vbs_vvh/ewkcoffea/analysis/vbs_vvh/histos/check_wgt_rwgtscan.pkl.gz"

    var_name = hist_to_use 
    #var_name = "njets_counts"
    cat = "exactly1lep_exactly1fj_STmet1100"

    #cat_yld = sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat}].values(flow=True)))
    #cat_err = (sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat}].variances(flow=True))))**0.5
    #print(cat_yld, cat_err)
    #exit()

    wgts = []
    for i in range(120):
        idx_name = f"idx{i}"
        cat_yld = sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat, "rwgtidx":idx_name}].values(flow=True)))
        cat_err = (sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat, "rwgtidx":idx_name}].variances(flow=True))))**0.5
        wgts.append(cat_yld)
        print(i,cat_yld)

    print(min(wgts))



### Dumps the yields and counts for a couple categories into a json ###
# The output of this is used for the CI check
def dump_json_simple(histo_dict, output_dir, out_name="vvh_yields_simple"):
    out_dict = {}
    cats_to_check = ["all_events", "exactly1lep_exactly1fj", "presel", "preselHFJ", "preselVFJ"]
    for proc_name in histo_dict[hist_to_use].axes["process"]:
        out_dict[proc_name] = {}
        for cat_name in cats_to_check:
            yld = sum(sum(histo_dict[hist_to_use][{"systematic":"nominal", "category":cat_name}].values(flow=True)))
            out_dict[proc_name][cat_name] = [yld,None]

    # Dump counts dict to json
    output_name = os.path.join(output_dir, f"{out_name}.json")
    with open(output_name,"w") as out_file: json.dump(out_dict, out_file, indent=4)
    print(f"\nSaved json file: {output_name}\n")



### Get the sig and bkg yields and print or dump to json ###
def print_yields(histo_dict, years_to_prepend, cat_lst, output_dir, roundat=None, print_counts=False, dump_to_json=True, quiet=False, out_name="yields"):

    # Get ahold of the yields
    yld_dict    = get_yields_per_cat(histo_dict,hist_to_use,years_to_prepend, cat_lst)
    counts_dict = None
    if print_counts:
        counts_dict = get_yields_per_cat(histo_dict,hist_to_use,years_to_prepend, cat_lst, use_counts=True)

    group_lst_order = GRP_DICT_FULL.keys() #SIGNAL_LST+['Background', 'ttbar', 'VV', 'Vjets', 'QCD', 'ST', 'ttX', 'VH', 'VVV']

    # Print to screen
    if not quiet:

        ### Print readably ###
        print("\n--- Yields ---")
        for cat in yld_dict:
            print(f"\n{cat}")
            bkg_yld, bkg_err = yld_dict[cat]["Background"]
            bkg_perr = 100*(bkg_err/bkg_yld)
            print(f"    Background:  {np.round(bkg_yld,roundat)} +- {np.round(bkg_perr,2)}%")
            for sig_group_name in SIGNAL_LST:                
                sig_yld, sig_err = yld_dict[cat][sig_group_name]
                sig_perr = 100*(sig_err/sig_yld)
                print(f"    {sig_group_name}:  {np.round(sig_yld,roundat)} +- {np.round(sig_perr,2)}%")
                print(f"      -> Metric: {np.round(yld_dict[cat]['metric_'+sig_group_name][0],3)}")
                print(f"      -> For copy pasting: python dump_toy_card.py {sig_yld} {bkg_yld}")


        ### Print csv, build op as an out string ###

        # Append the header
        out_str = ""
        header = "cat name"
        for proc_name in group_lst_order:
            header = header + f", {proc_name}"
        header = header + ", Background"  # Add Background column (sum of all non-signal)
        for sig_group_name in SIGNAL_LST:
            header = header + f", metric_{sig_group_name}"
        out_str = out_str + header

        # Appead a line for each category, with yields and metric
        for cat in yld_dict:
            line_str = cat
            for group_name in group_lst_order:
                yld, err = yld_dict[cat][group_name]
                perr = 100*(err/yld) if yld > 0 else 0
                line_str = line_str + f" , {np.round(yld,roundat)} ± {np.round(perr,2)}%"
            # Add Background column (sum of all non-signal backgrounds)
            bkg_yld, bkg_err = yld_dict[cat]["Background"]
            bkg_perr = 100*(bkg_err/bkg_yld) if bkg_yld > 0 else 0
            line_str = line_str + f" , {np.round(bkg_yld,roundat)} ± {np.round(bkg_perr,2)}%"
            # And also append the metrics for each signal
            for sig_group_name in SIGNAL_LST:
                metric = yld_dict[cat]['metric_'+sig_group_name][0]
                line_str = line_str + f" , {np.round(metric,3)}"
            # Append the string for this line to the out string
            out_str = out_str + f"\n{line_str}"

        # Print the out string to the screen
        print("\n\n--- Yields CSV formatted ---\n")
        print(out_str)

        # Save CSV to file
        csv_output_name = os.path.join(output_dir, f"{out_name}.csv")
        with open(csv_output_name, "w") as csv_file:
            csv_file.write(out_str)
        print(f"\nSaved CSV file: {csv_output_name}\n")


    # Dump directly to json
    if dump_to_json:
        out_dict = {"yields":yld_dict, "counts":counts_dict}
        output_name = os.path.join(output_dir, f"{out_name}.json")
        with open(output_name,"w") as out_file: json.dump(out_dict, out_file, indent=4)
        if not quiet:
            print("\n\n--- Yields json formatted ---")
            print(f"\nSaved json file: {output_name}\n")



### Make the plots ###
def make_plots(histo_dict, year_name_lst_to_prepend, cat_lst, output_dir, var_lst=None):

    grouping_dict = append_years(GRP_DICT_FULL,year_name_lst_to_prepend)

    #cat_lst = CAT_LST
    if var_lst is None:
        var_lst = histo_dict.keys()

    # Create figure ONCE outside the loop and reuse it (much faster than creating new figure each time)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(9,10),
        gridspec_kw={"height_ratios": (3, 1, 1, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07, right=0.65)
    axes = (ax1, ax2, ax3, ax4)

    for cat in cat_lst:
        print("\nCat:",cat)
        for var in var_lst:
            print("Var:",var)
            #if var not in ["njets","njets_counts","scalarptsum_lepmet"]: continue # TMP

            # No need for deepcopy - group() creates new hist, merge_overflow() does its own deepcopy
            histo = histo_dict[var][{"systematic":"nominal", "category":cat}]

            # Clean up a bit (rebin, regroup, and handle overflow)
            #if var not in ["njets","nleps","nbtagsl","nbtagsm","njets_counts","nleps_counts","nfatjets","njets_forward","njets_tot"]:
                #histo = plt_tools.rebin(histo,6)
            # redefine axis with process categories regrouped according to dict
            histo = plt_tools.group(histo,"process","process_grp",grouping_dict)
            histo = plt_tools.merge_overflow(histo)

            # Get list of available process groups in the histogram
            available_process_groups = [str(x) for x in histo.axes["process_grp"]]

            # Get one hist of just bkg and separate hists for each signal
            grp_names_bkg_lst = [grp for grp in grouping_dict.keys()
                                 if "Signal" not in grp and grp in available_process_groups]

            # Create a dictionary of signal histograms (only for signals present in the histogram)
            histo_sig_dict = {}
            for sig_group_name in SIGNAL_LST:
                if sig_group_name in available_process_groups:
                    histo_sig_dict[sig_group_name] = histo[{"process_grp":[sig_group_name]}]
                else:
                    print(f"WARNING: Signal group {sig_group_name} not found in histogram, skipping...")

            # sum all background groups together. Note: resulting "process_grp" axis has only one category, i.e. "Background"
            histo_bkg_sum = plt_tools.group(histo,"process_grp","process_grp",{"Background":grp_names_bkg_lst})

            # Get histogram with only background groups (for stacked plot)
            histo_bkg_only = histo[{"process_grp": grp_names_bkg_lst}]

            # Output dir
            save_dir_path = os.path.join(output_dir, "plots_sigbkg_scans")
            os.makedirs(save_dir_path, exist_ok=True)
            save_dir_path_cat = os.path.join(save_dir_path, cat)
            os.makedirs(save_dir_path_cat, exist_ok=True)

            # Custom axis ranges for specific variables (add more as needed)
            axis_ranges = {
                # "mjj_max": (0, 1500),
                "jj_pairs_atmindr_mjj": (0, 200),
            }
            axisrangex = axis_ranges.get(var, None)

            # Make the figure (reuse fig and axes for speed)
            log_y = False
            title = f"{cat}__{var}"
            if log_y:
                title+="_logY"
            fig, axes, ext_tup = make_vvh_fig(
                histo_mc = histo_bkg_only,
                histo_mc_sig_dict = histo_sig_dict,
                histo_mc_bkg = histo_bkg_sum,
                title=title,
                log_y=log_y,
                fig=fig,
                axes=axes,
                axisrangex=axisrangex
            )

            fig.savefig(os.path.join(save_dir_path_cat, title+".png"), bbox_extra_artists=ext_tup, bbox_inches='tight')

            # Make the same figure in log Y scale (reuse fig and axes)
            log_y = True
            title = f"{cat}__{var}"
            if log_y:
                title+="_logY"
            fig, axes, ext_tup = make_vvh_fig(
                histo_mc = histo_bkg_only,
                histo_mc_sig_dict = histo_sig_dict,
                histo_mc_bkg = histo_bkg_sum,
                title=title,
                log_y=log_y,
                fig=fig,
                axes=axes,
                axisrangex=axisrangex
            )

            fig.savefig(os.path.join(save_dir_path_cat, title+".png"), bbox_extra_artists=ext_tup, bbox_inches='tight')

            #shutil.copyfile(HTML_PC, os.path.join(save_dir_path_cat, "index.php"))

    # Close figure when done with all plots
    plt.close(fig)


### Print efficiency table from tagger efficiency cutflows ###
def print_efficiency_table(histo_dict, years_to_prepend, output_dir, out_name="efficiency_table"):
    """
    Print an efficiency table from the tagger efficiency cutflow.

    Expected categories (from 1FJMET_tagger_eff cutflow):
      - 1FJMET_tagger_eff_presel_1fj: baseline (presel & nfj_eq1)
      - 1FJMET_tagger_eff_truth_is_H: events where fj0 is truth-matched to H
      - 1FJMET_tagger_eff_truth_is_V: events where fj0 is truth-matched to V
      - 1FJMET_tagger_eff_tagger_correct_H: truth H AND tagger says H
      - 1FJMET_tagger_eff_tagger_correct_V: truth V AND tagger says V

    Efficiency_H = yield(tagger_correct_H) / yield(truth_is_H)
    Efficiency_V = yield(tagger_correct_V) / yield(truth_is_V)
    """
    print("\n" + "="*80)
    print("TAGGER EFFICIENCY TABLE")
    print("="*80)

    # Get the initial grouping dict (signals only for efficiency)
    grouping_dict = append_years(GRP_DICT_FULL, years_to_prepend)

    # Categories for efficiency calculation
    cat_mapping = {
        "presel_1fj": "1FJMET_tagger_eff_presel_1fj",
        "truth_is_H": "1FJMET_tagger_eff_truth_is_H",
        "truth_is_V": "1FJMET_tagger_eff_truth_is_V",
        "tagger_correct_H": "1FJMET_tagger_eff_tagger_correct_H",
        "tagger_correct_V": "1FJMET_tagger_eff_tagger_correct_V",
    }

    # Check which categories exist
    available_cats = list(histo_dict[hist_to_use].axes["category"])
    missing_cats = [v for v in cat_mapping.values() if v not in available_cats]
    if missing_cats:
        print(f"WARNING: Missing categories for efficiency calculation: {missing_cats}")
        print("Available categories:", available_cats)
        return

    # Results storage for table output
    results = []

    # Loop over signal samples
    for signal_name in SIGNAL_LST:
        if signal_name not in grouping_dict:
            continue

        signal_procs = grouping_dict[signal_name]

        # Get yields for each category
        yields = {}
        for short_name, full_cat in cat_mapping.items():
            histo_base = histo_dict[hist_to_use][{"systematic": "nominal", "category": full_cat}]
            histo = plt_tools.group(histo_base, "process", "process", {signal_name: signal_procs})
            yld = sum(sum(histo.values(flow=True)))
            var = sum(sum(histo.variances(flow=True)))
            yields[short_name] = (yld, np.sqrt(var))

        # Calculate efficiencies
        # Efficiency_H = tagger_correct_H / truth_is_H
        truth_H_yld, truth_H_err = yields["truth_is_H"]
        correct_H_yld, correct_H_err = yields["tagger_correct_H"]
        if truth_H_yld > 0:
            eff_H = correct_H_yld / truth_H_yld
            # Error propagation: eff_err = eff * sqrt((dc/c)^2 + (dt/t)^2)
            eff_H_err = eff_H * np.sqrt((correct_H_err/correct_H_yld)**2 + (truth_H_err/truth_H_yld)**2) if correct_H_yld > 0 else 0
        else:
            eff_H, eff_H_err = 0, 0

        # Efficiency_V = tagger_correct_V / truth_is_V
        truth_V_yld, truth_V_err = yields["truth_is_V"]
        correct_V_yld, correct_V_err = yields["tagger_correct_V"]
        if truth_V_yld > 0:
            eff_V = correct_V_yld / truth_V_yld
            eff_V_err = eff_V * np.sqrt((correct_V_err/correct_V_yld)**2 + (truth_V_err/truth_V_yld)**2) if correct_V_yld > 0 else 0
        else:
            eff_V, eff_V_err = 0, 0

        # Store results
        results.append({
            "signal": signal_name,
            "presel_1fj": yields["presel_1fj"][0],
            "truth_H": truth_H_yld,
            "truth_V": truth_V_yld,
            "correct_H": correct_H_yld,
            "correct_V": correct_V_yld,
            "eff_H": eff_H,
            "eff_H_err": eff_H_err,
            "eff_V": eff_V,
            "eff_V_err": eff_V_err,
        })

        # Print per-signal summary
        print(f"\n{signal_name}:")
        print(f"  Presel + 1FJ:     {yields['presel_1fj'][0]:.2f}")
        print(f"  Truth is H:       {truth_H_yld:.2f} ({100*truth_H_yld/yields['presel_1fj'][0]:.1f}% of presel)")
        print(f"  Truth is V:       {truth_V_yld:.2f} ({100*truth_V_yld/yields['presel_1fj'][0]:.1f}% of presel)")
        print(f"  Tagger correct H: {correct_H_yld:.2f}")
        print(f"  Tagger correct V: {correct_V_yld:.2f}")
        print(f"  Efficiency H:     {100*eff_H:.1f} +/- {100*eff_H_err:.1f}%")
        print(f"  Efficiency V:     {100*eff_V:.1f} +/- {100*eff_V_err:.1f}%")

    # Print summary table
    print("\n" + "-"*80)
    print("SUMMARY TABLE")
    print("-"*80)
    print(f"{'Signal':<35} {'Eff_H (%)':<15} {'Eff_V (%)':<15}")
    print("-"*80)
    for r in results:
        eff_H_str = f"{100*r['eff_H']:.1f} +/- {100*r['eff_H_err']:.1f}"
        eff_V_str = f"{100*r['eff_V']:.1f} +/- {100*r['eff_V_err']:.1f}"
        print(f"{r['signal']:<35} {eff_H_str:<15} {eff_V_str:<15}")
    print("-"*80)

    # Dump to JSON if output_dir provided
    if output_dir:
        out_dict = {r["signal"]: {
            "eff_H": r["eff_H"],
            "eff_H_err": r["eff_H_err"],
            "eff_V": r["eff_V"],
            "eff_V_err": r["eff_V_err"],
            "yields": {
                "presel_1fj": r["presel_1fj"],
                "truth_H": r["truth_H"],
                "truth_V": r["truth_V"],
                "correct_H": r["correct_H"],
                "correct_V": r["correct_V"],
            }
        } for r in results}

        output_path = os.path.join(output_dir, f"{out_name}.json")
        with open(output_path, "w") as f:
            json.dump(out_dict, f, indent=4)
        print(f"\nSaved efficiency table to: {output_path}")


##################################### Main #####################################

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    parser.add_argument('-y', "--get-yields", action='store_true', help = "Get yields from the pkl file")
    parser.add_argument('-p', "--make-plots", action='store_true', help = "Make plots from the pkl file")
    parser.add_argument('-j', "--dump-json", action='store_true', help = "Dump some yield numbers into a json file")
    parser.add_argument('-e', "--efficiency-table", action='store_true', help = "Print tagger efficiency table (requires 1FJMET_tagger_eff cutflow)")
    parser.add_argument('-o', "--output-name", default='vvh', help = "What to name the outputs")
    parser.add_argument("--outdir", default="output/", help="Output directory to store studies.")
    parser.add_argument("--projdir", default="proj_test/", help="Output sub-directory to store studies in separate project.")
    parser.add_argument("--use-variables", default=None, help="Comma-separated list of variables to plot (default: all).")
    parser.add_argument("--use-categories", default=None, help="Comma-separated list of categories (or cutflows) to plot (default: all). Supports wildcards.")
    args = parser.parse_args()

    # Compute output directory and create if it doesn't exist
    output_dir = os.path.join(args.outdir, args.projdir)
    os.makedirs(output_dir, exist_ok=True)

    # Get the dictionary of histograms from the input pkl file
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))

    from hist import Hist
    variables = [k for k in histo_dict if isinstance(histo_dict[k], Hist)]
    print(variables)
    first_h = histo_dict[variables[0]]

    # Parse variables and categories from args or use all available
    use_variables = args.use_variables.split(",") if args.use_variables else variables

    # Parse categories with glob/wildcard support
    all_categories = list(first_h.axes["category"])
    print(f"all_categories = {all_categories}")
    if args.use_categories:
        use_categories = []
        for pattern in args.use_categories.split(","):
            # Use fnmatch to support wildcards like 'all_events*'
            matched = fnmatch.filter(all_categories, pattern)
            if matched:
                use_categories.extend(matched)
            else:
                # If no match, maybe it's an exact category name, add it anyway
                if pattern in all_categories:
                    use_categories.append(pattern)
                else:
                    print(f"WARNING: Pattern '{pattern}' did not match any categories")
        # Remove duplicates while preserving order
        use_categories = list(dict.fromkeys(use_categories))
    else:
        use_categories = all_categories
    cat_lst = use_categories
    print(f"use_variables = {use_variables}")
    print(f"cat_lst = {cat_lst}")
    # Print total raw events
    #tot_raw = sum(sum(histo_dict["njets_counts"][{"systematic":"nominal", "category":"all_events"}].values(flow=True)))
    #print("Tot raw events:",tot_raw)
    #print(histo_dict["njets"])

    print("processes:")
    print(plt_tools.get_axis_cats(histo_dict[hist_to_use],"process"))

    # Figure out the proc naming convention
    proc_name = plt_tools.get_axis_cats(histo_dict[hist_to_use],"process")[0]
    if proc_name.startswith("UL"): years_to_prepend = ["UL16APV","UL16","UL17","UL18"] # Looks like ewkcoffea convention
    else: years_to_prepend = ["2016postVFP","2016preVFP","2017","2018"] # Otherwise from RDF convention

    # Which main functionalities to run
    if args.dump_json:
        dump_json_simple(histo_dict, output_dir, args.output_name)
    if args.get_yields:
        print_yields(histo_dict, years_to_prepend, cat_lst, output_dir, out_name=args.output_name+"_yields_sig_bkg", roundat=4, print_counts=False, dump_to_json=True)
    if args.make_plots:
        make_plots(histo_dict, years_to_prepend, cat_lst, output_dir, var_lst=use_variables)
    if args.efficiency_table:
        print_efficiency_table(histo_dict, years_to_prepend, output_dir, out_name=args.output_name+"_efficiency")


main()

