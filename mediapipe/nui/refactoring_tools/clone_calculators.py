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
import glob
import os
import re
import difflib
from collections import defaultdict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def index_mediapipe_simple_subgraphs(modules_root="mediapipe/modules"):
    """
    Index all mediapipe_simple_subgraph macros under mediapipe/modules.
    Returns a dict: register_as -> { 'bazel_target': '//path/to:target', 'deps': [...], 'build_path': ... }
    """
    subgraph_map = {}
    build_files = glob.glob(os.path.join(modules_root, "**/BUILD"), recursive=True)
    for build_path in build_files:
        with open(build_path, 'r') as f:
            content = f.read()
        # Find all mediapipe_simple_subgraph blocks
        for m in re.finditer(r"mediapipe_simple_subgraph\((.*?)\)\s*", content, re.DOTALL):
            block = m.group(1)
            name = None
            register_as = None
            # Parse name, register_as
            for line in block.splitlines():
                if 'name' in line:
                    name_match = re.search(r'name\s*=\s*"([^"]+)"', line)
                    if name_match:
                        name = name_match.group(1)
                if 'register_as' in line:
                    register_as_match = re.search(r'register_as\s*=\s*"([^"]+)"', line)
                    if register_as_match:
                        register_as = register_as_match.group(1)
            if name and register_as:
                rel_dir = os.path.relpath(os.path.dirname(build_path), os.getcwd())
                bazel_target = f"//{rel_dir}:{name}"
                subgraph_map[register_as] = {
                    'bazel_target': bazel_target,
                    'build_path': build_path
                }
    return subgraph_map

def main():
    parser = argparse.ArgumentParser(description='Clone a MediaPipe pipeline with a specified suffix.')
    parser.add_argument('--rewrite', type=str2bool, default=False, help='Whether to rewrite existing files (default: False)')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--single_calculator_only', help='Path to a single calculator source file to clone (absolute or relative)')
    input_group.add_argument('--pipeline_json', help='Path to the JSON file describing the pipeline hierarchy (absolute or relative) created by our pipeline parsing script. the same pipeline it was run for must be provided as the next argument value:')

    parser.add_argument('--pipeline_pbtxt_file', help='Path to the pbtxt file to clone (absolute or relative) which the provided --pipeline_json describes.')

    parser.add_argument('--clones_suffix', default="New", help='Suffix to add to the cloned calculators and graphs (default: "New")')

    parser.add_argument('--target_source_dir', default=os.path.join("mediapipe/nui/desktop/calculators"), help='Target directory for the cloned calculator and graph files (absolute or relative)')
    parser.add_argument('--target_build_file', default=os.path.join("mediapipe/nui/desktop/hand_tracking/BUILD"), help='Path to the build file to update for integrating the cloned entities (absolute or relative)')
    parser.add_argument('--calculators_build_file', default=os.path.join("mediapipe/nui/desktop/calculators/BUILD"), help='Path to the calculators BUILD file (absolute or relative) for integrating the cloned entities')
    parser.add_argument('--target_build_library_name', default="new", help='Name for the single cc_library integrating all cloned calculators (default: "new"). Using the same name as the suffix keeps thing simple, but you may use a more descriptive name here when applicable')

    parser.add_argument('--verbose', type=str2bool, default=False, help='Enable verbose output for debugging')
    args = parser.parse_args()

    all_cloned_src_files = set()
    nodes = []
    subgraph_map = index_mediapipe_simple_subgraphs()

    unmatched_subgraphs = []
    matched_subgraphs = []
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
                rel_path = os.path.relpath(dst_path, os.path.dirname(args.calculators_build_file))
                all_cloned_src_files.add(rel_path)
            # If this node is a graph, integrate its Bazel target only (not its deps)
            if node['type'].lower() == 'graph':
                reg_name = node['name']
                if reg_name in subgraph_map:
                    bazel_target = subgraph_map[reg_name]['bazel_target']
                    add_external_deps_to_build(args.calculators_build_file, [bazel_target], reg_name, args.verbose)
                    add_external_deps_to_build(args.target_build_file, [bazel_target], reg_name, args.verbose)
                    matched_subgraphs.append(reg_name)
                else:
                    unmatched_subgraphs.append(reg_name)
        if unmatched_subgraphs:
            print("\nWARNING: The following subgraphs do not have Bazel build definitions (mediapipe_simple_subgraph) in mediapipe/modules:")
            for s in unmatched_subgraphs:
                print(f"  - {s}")
                # Show closest matches for each unmatched subgraph
                close = difflib.get_close_matches(s, subgraph_map.keys(), n=3)
                if close:
                    print(f"    Closest matches: {', '.join(close)}")
            print("\nPlease check for typos or missing build definitions in mediapipe/modules.")
        if matched_subgraphs:
            print("\nThe following subgraphs were found and their Bazel build definitions will be integrated:")
            for s in matched_subgraphs:
                print(f"  - {s} (Bazel target: {subgraph_map[s]['bazel_target']})")
        if args.pipeline_pbtxt_file:
            pbtxt_dst = append_suffix_to_filename(args.pipeline_pbtxt_file, args.clones_suffix)
            clone_and_rename_file(args.pipeline_pbtxt_file, pbtxt_dst, pipeline['name'], pipeline['name'] + args.clones_suffix, name_map, args.rewrite, args.verbose, force_rename=True)
            # Ensure a build rule exists for the cloned pbtxt and add it as a dep
            pbtxt_base = os.path.splitext(os.path.basename(pbtxt_dst))[0]
            # Assume the build file for the pbtxt is calculators_build_file
            build_path = args.calculators_build_file
            with open(build_path, 'r') as f:
                build_content = f.read()
            rule_name = pbtxt_base
            bazel_target = f"//mediapipe/nui/desktop/calculators:{rule_name}"
            if f'name = "{rule_name}"' not in build_content:
                # Add a mediapipe_graph rule for the cloned pbtxt
                rule = f'''
mediapipe_graph(
    name = "{rule_name}",
    graph = "{os.path.basename(pbtxt_dst)}",
    visibility = ["//visibility:public"],
)
'''
                with open(build_path, 'a') as f:
                    f.write(rule)
                if args.verbose:
                    print(f"Added mediapipe_graph rule for {rule_name} to {build_path}")
            # Add the Bazel target to the deps in the target build file
            add_external_deps_to_build(args.target_build_file, [bazel_target], rule_name, args.verbose)
    elif args.single_calculator_only:
        src_path = args.single_calculator_only
        name = os.path.splitext(os.path.basename(src_path))[0]
        new_name = name + args.clones_suffix
        dst_path = os.path.join(args.target_source_dir, append_suffix_to_filename(os.path.basename(src_path), args.clones_suffix))
        clone_and_rename_file(src_path, dst_path, name, new_name, {}, args.rewrite, args.verbose, force_rename=True)
        if src_path.endswith('.cc') or src_path.endswith('.cpp'):
            rel_path = os.path.relpath(dst_path, os.path.dirname(args.calculators_build_file))
            all_cloned_src_files.add(rel_path)
    else:
        print("No valid input provided.")

    # Only add a single cc_library for all cloned calculators
    if all_cloned_src_files:
        add_cloned_library_to_build(args.calculators_build_file, args.target_build_library_name, sorted(all_cloned_src_files), args.verbose)
        add_cloned_library_dep_to_target_build_file(args.target_build_file, args.target_build_library_name, args.verbose)
        # --- FIX: Add all subgraph Bazel targets (at any depth) to the deps of the 'new' cc_library ---
        graph_targets = set()
        if args.pipeline_json:
            # Add the root pipeline Bazel target if present
            if args.pipeline_pbtxt_file:
                pbtxt_dst = append_suffix_to_filename(args.pipeline_pbtxt_file, args.clones_suffix)
                pbtxt_base = os.path.splitext(os.path.basename(pbtxt_dst))[0]
                graph_targets.add(f"//mediapipe/nui/desktop/calculators:{pbtxt_base}")
            # Add all subgraph Bazel targets from all nodes (recursively)
            for node in nodes:
                if node.get('type', '').lower() == 'graph':
                    reg_name = node['name']
                    if reg_name in subgraph_map:
                        graph_targets.add(subgraph_map[reg_name]['bazel_target'])
        if graph_targets:
            add_graph_targets_to_new_cc_library(args.calculators_build_file, args.target_build_library_name, list(graph_targets), args.verbose)

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
    Also, if src_path is a .cc file, clone and rename the corresponding .h file similarly.
    """
    if not os.path.exists(src_path):
        print(f"WARNING: source file {src_path} does not exist, skipping.")
        return
    if not rewrite and os.path.exists(dst_path):
        print(f"skipping target file {dst_path} because it already exists")
        return
    with open(src_path, 'r') as f:
        content = f.read()
    # Update #include statement for the header if this is a .cc file
    if src_path.endswith('.cc') or src_path.endswith('.cpp'):
        # Determine the new header file name and path
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        new_header_name = base_name + f'_{new_name[len(old_name):].lower()}.h'
        new_header_path = f'mediapipe/nui/desktop/calculators/{new_header_name}'
        # Replace any #include of the original header with the new one
        content = re.sub(r'#include\s+"[^"]*' + re.escape(base_name) + r'\.h"', f'#include "{new_header_path}"', content)
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
    # Ensure parent directory exists before writing
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        f.write(content)
    print(f"cloned & renamed {src_path} --> {dst_path}")
    if verbose:
        print(f"Cloned and renamed {src_path} -> {dst_path}")

    # If src_path is a .cc file, also clone and rename its .h file
    if src_path.endswith('.cc') or src_path.endswith('.cpp'):
        src_dir = os.path.dirname(src_path)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        src_h_path = os.path.join(src_dir, base_name + '.h')
        if os.path.exists(src_h_path):
            dst_h_path = os.path.join(os.path.dirname(dst_path), base_name + f'_{new_name[len(old_name):].lower()}.h')
            with open(src_h_path, 'r') as f:
                h_content = f.read()
            # Rename class name in .h file
            h_content = re.sub(rf'\b{re.escape(old_name)}\b', new_name, h_content)
            # Rename references to subnodes in .h file
            for sub_old, sub_new in subnode_renames.items():
                h_content = re.sub(rf'\b{sub_old}\b', sub_new, h_content)
            # Robust include guard update
            guard_match = re.search(r'#ifndef\s+([A-Z0-9_]+)', h_content)
            if guard_match:
                old_guard = guard_match.group(1)
                # Construct new guard from the new header path
                rel_path = os.path.relpath(dst_h_path, os.path.join(os.getcwd(), ''))
                new_guard_base = re.sub(r'[^A-Za-z0-9]', '_', rel_path).upper()
                # Remove trailing _H or _H_ from new_guard_base
                new_guard_base = re.sub(r'_H_?$', '', new_guard_base)
                new_guard = new_guard_base + '_H_'
                # Replace all instances of old_guard with new_guard
                h_content = re.sub(rf'(#ifndef\s+){old_guard}', rf'\1{new_guard}', h_content)
                h_content = re.sub(rf'(#define\s+){old_guard}', rf'\1{new_guard}', h_content)
                h_content = re.sub(rf'(#endif\s*//\s*){old_guard}', rf'\1{new_guard}', h_content)
                h_content = re.sub(rf'(#endif\s*)//\s*{old_guard}', rf'\1// {new_guard}', h_content)
            os.makedirs(os.path.dirname(dst_h_path), exist_ok=True)
            with open(dst_h_path, 'w') as f:
                f.write(h_content)
            print(f"cloned & renamed {src_h_path} --> {dst_h_path}")
            if verbose:
                print(f"Cloned and renamed {src_h_path} -> {dst_h_path}")

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
        # "//mediapipe/framework:calculator_framework",
        # "//mediapipe/framework:calculator_base",
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
    Add a single cc_library rule for all cloned calculators to the BUILD file, including their headers in hdrs if they exist.
    """
    with open(build_path, 'r') as f:
        build_content = f.read()
    if f'name = "{library_name}"' in build_content:
        if verbose:
            print(f"Library {library_name} already present in {build_path}")
        return
    hdrs = []
    for src in src_files:
        if src.endswith('.cc') or src.endswith('.cpp'):
            header_path = src[:-3] + '.h'  # Replace .cc with .h
            abs_header_path = os.path.join(os.path.dirname(build_path), header_path)
            if os.path.exists(abs_header_path):
                hdrs.append(header_path)
    hdrs_unique = sorted(set(hdrs))
    srcs_unique = sorted(set(src_files))
    srcs_list = ',\n        '.join([f'"{src}"' for src in srcs_unique])
    hdrs_list = ',\n        '.join([f'"{hdr}"' for hdr in hdrs_unique])
    rule = f'''
cc_library(
    name = "{library_name}",
    srcs = [
        {srcs_list}
    ],
    hdrs = [
        {hdrs_list}
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
    print(f"Integrated all cloned calculators and their headers into cc_library: {library_name}")
    if verbose:
        print(f"Added single build rule for {library_name} to {build_path}")

def add_cloned_library_dep_to_target_build_file(build_path, target_build_library_name, verbose=False):
    """
    Add the new cc_library as a dep in the cc_binary in the given BUILD file, avoiding duplicates.
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
            # Only insert if not already present
            if not any(f'"//mediapipe/nui/desktop:{target_build_library_name}"' in l for l in lines):
                new_lines.append(dep_line)
            inserted = True
    with open(build_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Added dependency {dep_line.strip()} to cc_binary in {build_path}")
    if verbose:
        print(f"Updated deps in cc_binary")

def add_external_deps_to_build(build_path, deps, target_name, verbose=False):
    """
    Add external Bazel deps to the cc_library or cc_binary named target_name in the given BUILD file, avoiding duplicates.
    If the deps field does not exist, create it.
    """
    with open(build_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    in_target = False
    target_indent = ''
    found_name = False
    in_deps = False
    deps_inserted = False
    deps_set = set(deps)
    for i, line in enumerate(lines):
        # Detect start of cc_library or cc_binary
        if re.match(r'\s*(cc_library|cc_binary)\s*\(', line):
            in_target = True
            target_indent = re.match(r'(\s*)', line).group(1)
            found_name = False
        if in_target and not found_name and f'name = "{target_name}"' in line:
            found_name = True
        # If inside the correct target, look for deps
        if in_target and found_name:
            if 'deps = [' in line:
                in_deps = True
                new_lines.append(line)
                continue
            if in_deps:
                if '],' in line:
                    # Insert new deps before closing ]
                    for dep in sorted(deps_set):
                        dep_line = f'{target_indent}    "{dep}",\n'
                        if dep_line not in lines and dep_line not in new_lines:
                            new_lines.append(dep_line)
                    in_deps = False
                    deps_inserted = True
            # If end of target and no deps were found, insert deps before closing paren
            if not in_deps and not deps_inserted and re.match(rf'{target_indent}\)', line):
                # Insert deps field
                new_lines.append(f'{target_indent}    deps = [\n')
                for dep in sorted(deps_set):
                    new_lines.append(f'{target_indent}        "{dep}",\n')
                new_lines.append(f'{target_indent}    ],\n')
                deps_inserted = True
        # Detect end of target
        if in_target and re.match(rf'{target_indent}\)', line):
            in_target = False
            found_name = False
            in_deps = False
        new_lines.append(line)
    with open(build_path, 'w') as f:
        f.writelines(new_lines)
    if verbose:
        print(f"Added external deps to {build_path} for target {target_name}: {deps}")

def add_graph_targets_to_new_cc_library(build_path, library_name, graph_targets, verbose=False):
    """
    Add Bazel targets for all cloned graphs to the deps of the cc_library named 'library_name' in the given BUILD file.
    """
    with open(build_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    in_new_lib = False
    in_deps = False
    for line in lines:
        # Detect start of the cc_library rule for 'library_name'
        if f'name = "{library_name}"' in line:
            in_new_lib = True
        if in_new_lib and 'deps = [' in line:
            in_deps = True
            new_lines.append(line)
            continue
        if in_deps:
            if '],' in line:
                # Insert new graph targets before closing ]
                for dep in sorted(set(graph_targets)):
                    dep_line = f'        "{dep}",\n'
                    if dep_line not in lines and dep_line not in new_lines:
                        new_lines.append(dep_line)
                in_deps = False
        new_lines.append(line)
        # End of cc_library rule
        if in_new_lib and line.strip() == ')':
            in_new_lib = False
    with open(build_path, 'w') as f:
        f.writelines(new_lines)
    if verbose:
        print(f"Added graph targets to deps of cc_library '{library_name}' in {build_path}: {graph_targets}")

if __name__ == "__main__":
    main()
