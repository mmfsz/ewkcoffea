import argparse
import pickle
import gzip
import json
import csv
import os
import numpy as np
import ewkcoffea.modules.plotting_tools as plt_tools
from hist import Hist  # Assuming hist library is available; adjust import if using coffea.hist
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import shutil
import warnings
warnings.filterwarnings("ignore", message="List indexing selection is experimental")


HTML_PC = "/home/users/kmohrman/ref_scripts/html_stuff/index.php"

CLR_LST = ['#d55e00', '#e69f00', '#f0e442', '#009e73', '#0072b2', '#56b4e9', '#cc79a7', '#6e3600', '#a17500'] 


def extract_yields(histo_dict, output_dir, hist_name="met", categories=None, output_csv=None, output_json=None):
    """
    Extract yields (weighted sums and uncertainties) from the histo_dict for each process and category.
    Outputs to CSV and/or JSON if specified.
    
    :param histo_dict: The loaded Coffea dict_accumulator containing histograms.
    :param hist_name: Name of the histogram to use for yield calculation (should be filled for all events).
    :param categories: List of categories to include; if None, use all from the axis.
    :param output_csv: Boolean to trigger CSV output (optional).
    :param output_json: Boolean to trigger JSON output (optional).
    :return: Dict of yields {process: {category: [value, uncertainty]}}
    """
    print("--> extract_yields()")
    if hist_name not in histo_dict:
        raise ValueError(f"Histogram '{hist_name}' not found in histo_dict.")
    else: 
        print(f"  Using histogram {hist_name} to obtain yields")
    h = histo_dict[hist_name]
    
    processes = list(h.axes["process"])
    if categories is None:
        categories = list(h.axes["category"])
    
    yields = {}
    for proc in processes:
        yields[proc] = {}
        for cat in categories:
            proj_h = h[{"process": proc, "category": cat, "systematic": "nominal"}]
            val = np.sum(proj_h.values(flow=True))
            var = np.sum(proj_h.variances(flow=True))
            unc = np.sqrt(var) if var > 0 else 0.0
            yields[proc][cat] = [val, unc]
    
    # Output to JSON if specified
    if output_json:
        json_path = os.path.join(output_dir, "output_yields.json")
        with open(json_path, "w") as f:
            json.dump(yields, f, indent=4)
        print(f"Saved JSON file: {json_path}")
    
    # Output to CSV if specified
    if output_csv:
        csv_path = os.path.join(output_dir, "output_yields.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header: Category, then each process
            header = ["Category"] + processes
            writer.writerow(header)
            for cat in categories:
                row = [cat] + [f"{yields[p][cat][0]:.4f} ± {yields[p][cat][1]:.4f}" for p in processes]
                writer.writerow(row)
        print(f"Saved CSV file: {csv_path}")
    
    return yields
import textwrap

def wrap_text(text, width):
    # Split the text into lines at the existing line breaks
    lines = text.split("\n")

    # Wrap each line independently
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=width))

    return "\n".join(wrapped_lines)

def draw_extra_text(h_counts, h_bins, text, logY=False, ax=None, fontsize=16):
    max_count = np.max(h_counts)
    min_count = np.min(h_counts)
    min_x_value = h_bins[0]
    text_x_position = min_x_value + (h_bins[-1] - min_x_value) * 0.02  # Slightly inset from left edge
    if logY:
        if min_count == 0: 
            min_count = 0.1        
        text_y_position = max_count * 5
        if ax:
            ax.set_yscale("log")
            ax.set_ylim(min_count, max_count * 10) 
        else:
            plt.yscale("log")
            plt.ylim(min_count, max_count * 10) 
    else: 
        if ax:
            ax.set_ylim(0, max_count * 1.1) 
        else:
            plt.ylim(0, max_count * 1.1) 
    
        text_y_position = max_count * 1.05

    # Use the wrapping function
    wrapped_text = wrap_text(text, width=40)  # width = number of characters
    
    if ax:
        ax.text(
            text_x_position, text_y_position, 
            wrapped_text, 
            fontsize=fontsize,
            ha='left',  # Horizontal alignment
            va='top'    # Vertical alignment
        )
    else:
        plt.text(
            text_x_position, text_y_position, 
            wrapped_text, 
            fontsize=fontsize,
            ha='left',  # Horizontal alignment
            va='top'    # Vertical alignment
        )

def plot_histograms(histo_dict, output_dir, variables=None, categories=None, systematic="nominal", groups=None,
                    density=False, logY=False, title=None, fontsize=14, leg_title=None,
                    drawCMSLabel=False, CoM=13.6, CMSFlag="Simulation Preliminary", extra_text=None, save_format="png", 
                    variables_def=None):
    """
    Plot histograms for specified variables and categories, with optional process grouping.
    
    :param histo_dict: The loaded Coffea dict_accumulator containing histograms.
    :param output_dir: Directory to save plots.
    :param variables: List of variable names to plot; if None, plot all valid Hist keys.
    :param categories: List of categories to plot; if None, use all from the axis.
    :param systematic: Systematic variation to use (default: 'nominal').
    :param groups: Dict of {group_name: [proc1, proc2, ...]}; if None, plot individual processes.
    :param density: If True, normalize to density.
    :param logY: If True, use log scale on y-axis.
    :param title: Optional plot title.
    :param fontsize: Font size for title, legend, extra text.
    :param leg_title: Legend title.
    :param drawCMSLabel: If True, draw CMS label.
    :param CoM: Center-of-mass energy for label.
    :param CMSFlag: CMS label flag (e.g., "Simulation Preliminary").
    :param extra_text: Optional extra text to draw (e.g., category info).
    :param variables_def: Dict of {var_name: Variable} for custom settings (e.g., xlabel, logY).
    """
    print(f"variables = {variables}")
    if variables is None:
        variables = [k for k in histo_dict if isinstance(histo_dict[k], Hist)]
    
    if len(variables) == 0:
        print("No variables to plot.")
        return
    
    #first_h = histo_dict[variables[0]]
    first_h = histo_dict["met_pt"]
    
    if categories is None:
        categories = list(first_h.axes["category"])
    print(f"categories = {categories}")
    
    all_processes = list(first_h.axes["process"])
    print(f"all_processes = {all_processes}")
    
    if groups is None:
        groups = {p: [p] for p in all_processes}

    # Create figure ONCE outside the loop and reuse it
    fig, ax = plt.subplots(figsize=(8, 8))

    for cat in categories:
        for var in variables:
            if var not in histo_dict:
                print(f"Skipping missing histogram: {var}")
                continue
            else: 
                print(f"Retrieving histogram {var}")
            
            h = histo_dict[var]
            
            # Check if the histogram has the expected axes
            if "process" not in h.axes.name or "category" not in h.axes.name or "systematic" not in h.axes.name:
                print(f"Skipping histogram {var} without required axes")
                continue
            
            # Apply custom Variable settings if provided
            custom_logY = logY
            custom_xlabel = None
            custom_extraTag = None
            if variables_def and var in variables_def:
                var_def = variables_def[var]
                custom_xlabel = var_def.xlabel
                custom_logY = var_def.logY
                if var_def.extraTag:
                    custom_extraTag = var_def.extraTag

            # Clear the axes for reuse (much faster than creating new figure)
            ax.clear()

            h_counts = []  # To collect counts for draw_extra_text
            h_bins = None  # To collect bins

            # Slice category and systematic ONCE outside the group loop
            h_cat = h[{"category": cat, "systematic": systematic}]
            var_axis_name = h_cat.axes[-1].name

            for group_name, procs in groups.items():
                # Filter to existing processes
                existing_procs = [p for p in procs if p in all_processes]
                if not existing_procs:
                    continue

                # Select all processes at once and project (sums over process axis)
                h_group = h_cat[{"process": existing_procs}].project(var_axis_name)

                # Skip first bin (often contains spike at 0 for non-existent objects)
                if "truth" in var_axis_name:
                    h_group = h_group[1:]

                # Plot with step style and collect data for text positioning
                h_group.plot(ax=ax, label=group_name, histtype="step", linewidth=2, density=density)
                # Extract year from the first process (assuming all processes in the group share the same year)
                # if existing_procs:
                #     year = existing_procs[0].split('_')[0]
                #     h_group = h_group[{"year": year}]
                    
                if h_bins is None:
                    h_bins = h_group.axes[0].edges  # Get bins from first hist
                h_counts.append(h_group.values())  # Collect counts per group
            
            h_counts = np.concatenate(h_counts) if h_counts else np.array([0])  # Flatten for max/min
            
            if ax.get_legend_handles_labels()[0]:  # Only add legend if there are plots
                ax.legend(title=leg_title, fontsize=fontsize)
            
            #if title:
            #    ax.set_title(title, fontsize=fontsize)
            ax.set_title(cat, fontsize=fontsize)
            
            # Set labels
            var_axis = h.axes[-1]
            xlabel = custom_xlabel or var_axis.label or var_axis.name
            ax.set_xlabel(xlabel)
            ylabel = "Density" if density else "Events"
            ax.set_ylabel(ylabel)

            if custom_logY:
                ax.set_yscale("log")
            
            if logY:
                ax.set_yscale("log")
            
            # CMS label
            if drawCMSLabel:
                data_flag = False if "Simulation" in CMSFlag and CMSFlag !="" else True
                hep.cms.label(
                    CMSFlag,
                    data=data_flag,
                    loc=0,
                    ax=ax,
                    rlabel=f"$\\sqrt{{s}} = {CoM}$ TeV",
                )
            
            # Draw extra text (e.g., category)
            if extra_text:
                draw_extra_text(h_counts, h_bins, extra_text, logY=custom_logY, ax=ax, fontsize=fontsize)
            # elif cat:  # Default to showing category if no extra_text provided
            #     draw_extra_text(h_counts, h_bins, f"{cat}", logY=logY, ax=ax, fontsize=fontsize)
            
            # Save with optional suffixes
            plot_name = f"{var}_{cat}"
            if density:
                plot_name += "_norm"
            if logY:
                plot_name += "_logY"
            if custom_extraTag:
                plot_name += f"_{custom_extraTag}"

            # Save
            save_dir = os.path.join(output_dir,cat)
            os.makedirs(save_dir, exist_ok=True)
            
            plot_path = os.path.join(save_dir, f"{plot_name}.{save_format}")
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Saved plot: {plot_path}")

    # Close figure when done with all plots
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Process Coffea histogram pickle file.")
    parser.add_argument("input_pkl", help="Path to the input pkl.gz file.")
    parser.add_argument("-p", "--hist-placeholder", default="met_pt", help="Histogram name to use as placeholder for yields (default: 'met_pt').")
    parser.add_argument("--outdir", default="output/", help="Output directory to store studies.")
    parser.add_argument("--projdir", default="proj_test/", help="Output sub-directory to store studies in separate project.")
    parser.add_argument("--output-csv", action='store_true', help="Enable CSV output for yields.")
    parser.add_argument("--output-json", action='store_true', help="Enable JSON output for yields.")
    parser.add_argument("--do-plots", action='store_true', help="Enable plotting of histograms.")
    parser.add_argument("--do-yields", action='store_true', help="Extract yield table (turn corresponding flags on to store to csv or json).")
    parser.add_argument("--use-variables", default=None, help="Comma-separated list of variables to plot (default: all).")
    parser.add_argument("--use-categories", default=None, help="Comma-separated list of categories (or cutflows) to plot (default: all).")
    parser.add_argument("--process-groups", default=None, help="Path to JSON file defining process groups {group_name: [proc1, proc2, ...]}.")
    parser.add_argument("--variables-def", default=None, help="Path to python file defining variables_dict = {name: Variable(...)}.")
    
    args = parser.parse_args()
    
    with gzip.open(args.input_pkl, "rb") as f:
        histo_dict = pickle.load(f)
    
    print(f"Loaded input file {args.input_pkl}")
    
    # Compute output directory
    output_dir = os.path.join(args.outdir, args.projdir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    
    use_variables = args.use_variables.split(",") if args.use_variables else None
    use_categories = args.use_categories.split(",") if args.use_categories else None
    print("==="*10)
    print(f" Variables")
    print(f"   {use_variables}")
    print(f" Categories")
    print(f"    {use_categories}")

    groups = None
    if args.process_groups:
        with open(args.process_groups, "r") as f:
                groups = json.load(f)
        print(f" Groups: ")
        print(f"   {groups}")
    print("==="*10)


   
    if args.do_yields:
        if not args.output_json and not args.output_csv:
            args.output_json = True
        extract_yields(histo_dict, output_dir, hist_name=args.hist_placeholder, categories=use_categories,
                       output_csv=args.output_csv, output_json=args.output_json)
    
    if args.do_plots:
        plot_histograms(histo_dict, output_dir, variables=use_variables, categories=use_categories, groups=groups)

if __name__ == "__main__":
    main()