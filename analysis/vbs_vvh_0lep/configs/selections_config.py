import yaml
import awkward as ak
import re

def get_cutflow(yaml_file, cutflow=None):
    """
    Load cutflow from a YAML file with atomic cuts and mode.
    Returns dict of {cf_name: (mode, step_dict)} if cutflow is None.
    """
    
    def make_lambda(expr: str):
        return eval(f"lambda events, var, objects: {expr}", {"ak": ak})
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Load atomic cuts (as strings)
    cuts_dict = config.get('cuts', {})

    def resolve_expr(raw_expr: str, visited=None):
        """Recursively resolve cut references until no more remain."""
        if visited is None:
            visited = set()

        resolved = raw_expr
        changed = True
        while changed:
            changed = False
            for cut_name, cut_expr in cuts_dict.items():
                if cut_name in visited:
                    continue
                # Use word boundary to match whole cut names only
                pattern = r'\b' + re.escape(cut_name) + r'\b'
                if re.search(pattern, resolved):
                    visited.add(cut_name)
                    # Wrap in parentheses to preserve operator precedence
                    resolved = re.sub(pattern, f'({cut_expr})', resolved)
                    changed = True
        return resolved

    cutflow_dict = {}

    for cf_name, cf_config in config.get('cutflows', {}).items():
        # Extract mode (default 'cumulative')
        mode = cf_config.pop('mode', 'cumulative')
        
        step_dict = {}
        for step, raw_expr in cf_config.items():
            resolved_expr = resolve_expr(raw_expr)
            step_dict[step] = make_lambda(resolved_expr)
        
        cutflow_dict[cf_name] = (mode, step_dict)

    if cutflow is None:
        print(f"read cutflow from project: {yaml_file}")
        return cutflow_dict
    else:
        print(f"read cutflow from project: {yaml_file}/{cutflow}")
        return cutflow_dict[cutflow]
    