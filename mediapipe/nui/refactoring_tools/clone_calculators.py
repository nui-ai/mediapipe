"""
This script clones a given mediapipe pipeline given a json file describing the pipeline hierarchy to avoid this script
having to do that itself, into a given target directory, while renaming all graphs and calculators and sub-graphs
recursively called by it, to the given suffix, both their file names and their internal names. for file names,
append _lowercase(suffix) to the name, for class names, append suffix to the name as given.

the arguments and their default values are already defined below.

to make the resulting cloned entire pipeline correct, each mention in each source file, of any of its subnodes,
should refer to that node by its new name too.

the root pipeline given as argument (which the json is already describing in full) should also be cloned and
renamed just the same, and the recursion implementing it all should be correct.

each cloned calculator (not graph) is also made to build same as TickCountCalculator is, by modifying the BUILD files.

"""

import argparse
import json
import os
import re
import shutil




def main():
    parser = argparse.ArgumentParser(description='Clone a MediaPipe pipeline with a specified suffix.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pipeline_json', help='Path to the JSON file describing the pipeline hierarchy (absolute or relative)')
    input_group.add_argument('--single_calculator', help='Path to a single calculator source file to clone (absolute or relative)')
    parser.add_argument('--pbtxt_file', help='Path to the pbtxt file to clone (absolute or relative)')
    parser.add_argument('--suffix', default="New", help='Suffix to add to calculator names (default: "New")')
    parser.add_argument('--target_dir', default=os.path.join("mediapipe/nui/desktop/calculators"), help='Target directory for cloned calculator files (absolute or relative)')
    parser.add_argument('--calculators_build_path', default=os.path.join("mediapipe/nui/desktop/calculators/BUILD"), help='Path to the calculators BUILD file (absolute or relative)')
    parser.add_argument('--hand_tracking_build_path', default=os.path.join("mediapipe/nui/desktop/hand_tracking/BUILD"), help='Path to the hand tracking BUILD file (absolute or relative)')
    parser.add_argument('--rewrite', action='store_true', default=False, help='Whether to rewrite existing files (default: False)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    args = parser.parse_args()

    if args.pipeline_json:
        with open(args.pipeline_json, 'r') as f:
            pipeline = json.load(f)
        nodes = get_all_nodes(pipeline, args.suffix)
        # Build subnode rename map for each node
        name_map = {n['name']: n['new_name'] for n in nodes}
        for node in nodes:
            src_path = node['source']
            dst_path = os.path.join(args.target_dir, os.path.basename(node['new_source']))
            subnode_renames = {sub['name']: sub['new_name'] for sub in nodes if sub['name'] != node['name']}
            force_rename = node.get('type', None) == 'calculator'
            clone_and_rename_file(src_path, dst_path, node['name'], node['new_name'], subnode_renames, args.rewrite, args.verbose, force_rename=force_rename)
            # Only add calculators (not graphs/pbtxt) to BUILD
            if src_path.endswith('.cc') or src_path.endswith('.cpp'):
                add_calculator_to_build(args.calculators_build_path, node['new_name'].lower(), os.path.relpath(dst_path, os.path.dirname(args.calculators_build_path)), args.verbose)
        # Clone and update pbtxt file
        if args.pbtxt_file:
            pbtxt_dst = append_suffix_to_filename(args.pbtxt_file, args.suffix)
            clone_and_rename_file(args.pbtxt_file, pbtxt_dst, pipeline['name'], pipeline['name'] + args.suffix, name_map, args.rewrite, args.verbose, force_rename=True)
    elif args.single_calculator:
        src_path = args.single_calculator
        name = os.path.splitext(os.path.basename(src_path))[0]
        new_name = name + args.suffix
        dst_path = os.path.join(args.target_dir, append_suffix_to_filename(os.path.basename(src_path), args.suffix))
        clone_and_rename_file(src_path, dst_path, name, new_name, {}, args.rewrite, args.verbose, force_rename=True)
        add_calculator_to_build(args.calculators_build_path, new_name.lower(), os.path.relpath(dst_path, os.path.dirname(args.calculators_build_path)), args.verbose)
    else:
        print("No valid input provided.")

def get_all_nodes(node, suffix, nodes=None):
    """Recursively collect all calculators/graphs/sub-graphs from the pipeline JSON."""
    if nodes is None:
        nodes = []
    node_copy = dict(node)
    node_copy['new_name'] = node['name'] + suffix
    node_copy['new_source'] = append_suffix_to_filename(node['source'], suffix)
    nodes.append(node_copy)
    for subnode in node.get('nodes', []):
        get_all_nodes(subnode, suffix, nodes)
    return nodes

def append_suffix_to_filename(filepath, suffix):
    """Append _lowercase(suffix) before file extension in filename."""
    dirname, basename = os.path.split(filepath)
    name, ext = os.path.splitext(basename)
    return os.path.join(dirname, f"{name}_{suffix.lower()}{ext}")

def clone_and_rename_file(src_path, dst_path, old_name, new_name, subnode_renames, rewrite=False, verbose=False, force_rename=False):
    """
    Clone src_path to dst_path, renaming class names and references from old_name to new_name,
    and updating references to subnodes using subnode_renames dict {old: new}.
    If force_rename is True, always rename old_name to new_name everywhere in the file.
    """
    if not rewrite and os.path.exists(dst_path):
        if verbose:
            print(f"Skipping {dst_path}, already exists.")
        return
    with open(src_path, 'r') as f:
        content = f.read()
    # Always rename calculator node name everywhere if force_rename
    if force_rename:
        content = re.sub(rf'\b{re.escape(old_name)}\b', new_name, content)
    else:
        # Rename class names and calculator/graph names (robust for C++/pbtxt)
        content = re.sub(rf'(class|struct|calculator|graph|Calculator|Graph|CALCULATOR|GRAPH|name\s*=\s*")({re.escape(old_name)})',
                         lambda m: m.group(1) + m.group(2) + new_name[len(old_name):], content)
    # Rename references to subnodes
    for sub_old, sub_new in subnode_renames.items():
        content = re.sub(rf'\b{sub_old}\b', sub_new, content)
    with open(dst_path, 'w') as f:
        f.write(content)
    if verbose:
        print(f"Cloned and renamed {src_path} -> {dst_path}")

def add_calculator_to_build(build_path, calculator_name, src_file, verbose=False):
    """
    Add a cc_library rule for the new calculator to the BUILD file, following TickCountCalculator's pattern.
    """
    with open(build_path, 'r') as f:
        build_content = f.read()
    # Check if already present
    if f'"{calculator_name}"' in build_content:
        if verbose:
            print(f"{calculator_name} already present in {build_path}")
        return
    # Template based on TickCountCalculator
    rule = f'''
cc_library(
    name = "{calculator_name}",
    srcs = ["{src_file}"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_base",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
'''
    with open(build_path, 'a') as f:
        f.write(rule)
    if verbose:
        print(f"Added build rule for {calculator_name} to {build_path}")

if __name__ == "__main__":
    main()
