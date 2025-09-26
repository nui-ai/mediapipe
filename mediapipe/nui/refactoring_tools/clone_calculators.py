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
    parser.add_argument('--rewrite', action='store_true', default=False, help='Whether to rewrite existing files (default: False)')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--single_calculator_only', help='Path to a single calculator source file to clone (absolute or relative)')
    input_group.add_argument('--pipeline_json', help='Path to the JSON file describing the pipeline hierarchy (absolute or relative) created by our pipeline parsing script. the same pipeline it was run for must be provided as the next argument value:')

    parser.add_argument('--pipeline_pbtxt_file', help='Path to the pbtxt file to clone (absolute or relative) which the provided --pipeline_json describes.')

    parser.add_argument('--clones_suffix', default="New", help='Suffix to add to the cloned calculators and graphs (default: "New")')

    parser.add_argument('--target_source_dir', default=os.path.join("mediapipe/nui/desktop/calculators"), help='Target directory for the cloned calculator and graph files (absolute or relative)')
    parser.add_argument('--target_build_file', default=os.path.join("mediapipe/nui/desktop/hand_tracking/BUILD"), help='Path to the build file to update for integrating the cloned entities (absolute or relative)')
    parser.add_argument('--calculators_build_file', default=os.path.join("mediapipe/nui/desktop/calculators/BUILD"), help='Path to the calculators BUILD file (absolute or relative) for integrating the cloned entities')
    parser.add_argument('--target_build_library_name', default="new", help='Name for the single cc_library integrating all cloned calculators (default: "new"). Using the same name as the suffix keeps thing simple, but you may use a more descriptive name here when applicable')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    args = parser.parse_args()

    all_cloned_src_files = []
    nodes = []

    if args.pipeline_json:
        with open(args.pipeline_json, 'r') as f:
            pipeline = json.load(f)
        nodes = get_all_nodes(pipeline, args.clones_suffix)
        name_map = {n['name']: n['new_name'] for n in nodes}
        for node in nodes:
            src_path = node['source']
            dst_path = os.path.join(args.target_source_dir, os.path.basename(node['new_source']))
            subnode_renames = {sub['name']: sub['new_name'] for sub in nodes if sub['name'] != node['name']}
            force_rename = node.get('type', None) == 'calculator'
            clone_and_rename_file(src_path, dst_path, node['name'], node['new_name'], subnode_renames, args.rewrite, args.verbose, force_rename=force_rename)
            if src_path.endswith('.cc') or src_path.endswith('.cpp'):
                all_cloned_src_files.append(os.path.relpath(dst_path, os.path.dirname(args.calculators_build_file)))
        if args.pipeline_pbtxt_file:
            pbtxt_dst = append_suffix_to_filename(args.pipeline_pbtxt_file, args.clones_suffix)
            clone_and_rename_file(args.pipeline_pbtxt_file, pbtxt_dst, pipeline['name'], pipeline['name'] + args.clones_suffix, name_map, args.rewrite, args.verbose, force_rename=True)
    elif args.single_calculator_only:
        src_path = args.single_calculator_only
        name = os.path.splitext(os.path.basename(src_path))[0]
        new_name = name + args.clones_suffix
        dst_path = os.path.join(args.target_source_dir, append_suffix_to_filename(os.path.basename(src_path), args.clones_suffix))
        clone_and_rename_file(src_path, dst_path, name, new_name, {}, args.rewrite, args.verbose, force_rename=True)
        if src_path.endswith('.cc') or src_path.endswith('.cpp'):
            all_cloned_src_files.append(os.path.relpath(dst_path, os.path.dirname(args.calculators_build_file)))
    else:
        print("No valid input provided.")

    # Only add a single cc_library for all cloned calculators
    if all_cloned_src_files:
        add_cloned_library_to_build(args.calculators_build_file, args.target_build_library_name, all_cloned_src_files, args.verbose)
        add_cloned_library_dep_to_target_build_file(args.target_build_file, args.target_build_library_name, args.verbose)

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
        print(f"skipping target file {dst_path} because it already exists")
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
    print(f"cloned & renamed {src_path} --> {dst_path}")
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

def add_cloned_library_to_build(build_path, library_name, src_files, verbose=False):
    """
    Add a single cc_library rule for all cloned calculators to the BUILD file.
    """
    with open(build_path, 'r') as f:
        build_content = f.read()
    if f'name = "{library_name}"' in build_content:
        if verbose:
            print(f"Library {library_name} already present in {build_path}")
        return
    srcs_list = ',\n        '.join([f'"{src}"' for src in src_files])
    rule = f'''
cc_library(
    name = "{library_name}",
    srcs = [
        {srcs_list}
    ],
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
    print(f"Integrated all cloned calculators into cc_library: {library_name}")
    if verbose:
        print(f"Added single build rule for {library_name} to {build_path}")

def add_cloned_library_dep_to_target_build_file(build_path, target_build_library_name, verbose=False):
    """
    Add the new cc_library as a dep in the cc_binary in the given BUILD file.
    """
    with open(build_path, 'r') as f:
        lines = f.readlines()
    dep_line = f'        "//mediapipe/nui/desktop:{target_build_library_name}",\n'
    already_present = any(dep_line.strip() == line.strip() for line in lines)
    if already_present:
        if verbose:
            print(f"Dependency {dep_line.strip()} already present in {build_path}")
        return
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if line.strip() == '"//mediapipe/nui/desktop:tick_count_calculator",' and not inserted:
            new_lines.append(dep_line)
            inserted = True
    with open(build_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Added dependency {dep_line.strip()} to cc_binary in {build_path}")
    if verbose:
        print(f"Updated deps in cc_binary")

if __name__ == "__main__":
    main()
