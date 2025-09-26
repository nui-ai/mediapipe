#!/usr/bin/env python3
"""
Clone MediaPipe calculators with a specified suffix.

This script:
1. Finds the source files for calculators used in a given pipeline
2. Creates copies of these files with a specified suffix
3. Changes calculator names in the code to have the same suffix
4. Updates BUILD files to include the new calculators
5. Creates a copy of the pipeline pbtxt file with calculator names updated
"""

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Set, Tuple

# MediaPipe repository root directory
MEDIAPIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def read_json_file(file_path: str) -> dict:
    """Read and parse a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_calculators_from_json(json_data: dict, calculators: List[dict] = None,
                                 visited: Set[str] = None) -> List[dict]:
    """
    Recursively extract calculators from the pipeline JSON file.
    Returns a list of dictionaries with calculator info.
    """
    if calculators is None:
        calculators = []
    if visited is None:
        visited = set()

    # Skip if this node has already been visited
    if json_data.get('name') in visited:
        return calculators

    # Add this calculator if it has a source file
    if json_data.get('source') and json_data.get('type') == 'calculator':
        calculator_info = {
            'name': json_data.get('name'),
            'source': json_data.get('source')
        }
        calculators.append(calculator_info)
        visited.add(json_data.get('name'))

    # Recursively process child nodes
    for node in json_data.get('nodes', []):
        extract_calculators_from_json(node, calculators, visited)

    return calculators


def clone_calculator_file(source_path: str, suffix: str) -> str:
    """
    Clone a calculator source file with the given suffix.
    Returns the path to the new file.
    """
    if not os.path.exists(source_path):
        print(f"Warning: Source file not found: {source_path}")
        return None

    # Generate the new file path in the desktop/calculators directory
    filename = os.path.basename(source_path)
    basename, ext = os.path.splitext(filename)
    new_basename = f"{basename}_{suffix.lower()}"
    new_filename = f"{new_basename}{ext}"
    
    # Create the target directory in mediapipe/nui/desktop/calculators
    target_dir = os.path.join(MEDIAPIPE_ROOT, "mediapipe/nui/desktop/calculators")
    os.makedirs(target_dir, exist_ok=True)
    
    new_file_path = os.path.join(target_dir, new_filename)

    # Read the content of the original file
    with open(source_path, 'r') as file:
        content = file.read()

    # Find the calculator class name
    class_match = re.search(r'class\s+(\w+)\s*:', content)
    if not class_match:
        print(f"Warning: Could not find calculator class in {source_path}")
        return None

    original_class_name = class_match.group(1)
    new_class_name = f"{original_class_name}{suffix}"

    # Replace the class name and REGISTER_CALCULATOR call
    content = content.replace(f"class {original_class_name}", f"class {new_class_name}")
    content = content.replace(f"REGISTER_CALCULATOR({original_class_name})", f"REGISTER_CALCULATOR({new_class_name})")
    
    # Update any references to the calculator name in comments or examples
    content = content.replace(f"calculator: \"{original_class_name}\"", f"calculator: \"{new_class_name}\"")

    # Write the modified content to the new file
    with open(new_file_path, 'w') as file:
        file.write(content)

    print(f"Created {new_file_path}")
    return new_file_path


def update_build_file(calculators: List[dict], suffix: str) -> None:
    """
    Create or update the BUILD file for the cloned calculators.
    """
    build_file_path = os.path.join(MEDIAPIPE_ROOT, "mediapipe/nui/desktop/calculators/BUILD")
    
    # Check if BUILD file exists, create it if not
    if not os.path.exists(build_file_path):
        with open(build_file_path, 'w') as file:
            file.write("# BUILD file for cloned calculators\n\n")
    
    # Read existing BUILD file content
    with open(build_file_path, 'r') as file:
        build_content = file.read()
    
    # Create a list of all calculator targets we need to add
    calculator_targets = []
    for calc in calculators:
        if calc.get('source'):
            filename = os.path.basename(calc.get('source'))
            basename, _ = os.path.splitext(filename)
            target_name = f"{basename}_{suffix.lower()}"
            calculator_targets.append(target_name)
    
    # Create the aggregator cc_library target if it doesn't exist
    aggregator_name = f"cloned_calculators_{suffix.lower()}"
    if aggregator_name not in build_content:
        deps_list = [f"\":{target}\"," for target in calculator_targets]
        deps_str = "\n        ".join(deps_list)
        
        aggregator_lib = f"""
cc_library(
    name = "{aggregator_name}",
    visibility = ["//visibility:public"],
    deps = [
        {deps_str}
    ],
)
"""
        build_content += aggregator_lib
    
    # Add individual cc_library targets for each calculator if they don't exist
    for calc in calculators:
        if not calc.get('source'):
            continue
            
        filename = os.path.basename(calc.get('source'))
        basename, ext = os.path.splitext(filename)
        target_name = f"{basename}_{suffix.lower()}"
        source_name = f"{basename}_{suffix.lower()}{ext}"
        
        if target_name not in build_content:
            # Create basic library definition with common dependencies
            calculator_lib = f"""
cc_library(
    name = "{target_name}",
    srcs = ["{source_name}"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_base",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,  # Ensures the calculator registration code is always linked
)
"""
            build_content += calculator_lib
    
    # Write the updated BUILD file
    with open(build_file_path, 'w') as file:
        file.write(build_content)
    
    print(f"Updated {build_file_path}")


def update_hand_tracking_build(suffix: str) -> None:
    """
    Update the hand tracking BUILD file to include the cloned calculators.
    """
    build_file_path = os.path.join(MEDIAPIPE_ROOT, "mediapipe/nui/desktop/hand_tracking/BUILD")
    
    if not os.path.exists(build_file_path):
        print(f"Warning: Hand tracking BUILD file not found at {build_file_path}")
        return
    
    with open(build_file_path, 'r') as file:
        build_content = file.read()
    
    # Create a new target for the cloned calculators
    new_target_name = f"hand_tracking_tflite_{suffix.lower()}"
    aggregator_name = f"cloned_calculators_{suffix.lower()}"
    
    if new_target_name not in build_content:
        # Find the dependencies section of the original target
        deps_pattern = r'(name\s*=\s*"hand_tracking_tflite".*?deps\s*=\s*\[)(.*?)(\s*\])'
        match = re.search(deps_pattern, build_content, re.DOTALL)
        
        if match:
            # Create a new target with the same dependencies plus our cloned calculators
            original_deps = match.group(2)
            new_target = f"""
cc_binary(
    name = "{new_target_name}",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [{original_deps}
        "//mediapipe/nui/desktop/calculators:{aggregator_name}",
    ],
)
"""
            build_content += new_target
            
            # Write the updated BUILD file
            with open(build_file_path, 'w') as file:
                file.write(build_content)
            
            print(f"Added target {new_target_name} to {build_file_path}")
        else:
            print(f"Warning: Could not find dependencies in {build_file_path}")


def clone_pbtxt_file(original_pbtxt: str, suffix: str, calculators: List[dict]) -> str:
    """
    Clone a pbtxt file, replacing calculator names with their suffixed versions.
    Returns the path to the new file.
    """
    if not os.path.exists(original_pbtxt):
        print(f"Warning: Original pbtxt file not found: {original_pbtxt}")
        return None
    
    # Generate the new file path
    dirname = os.path.dirname(original_pbtxt)
    filename = os.path.basename(original_pbtxt)
    basename, ext = os.path.splitext(filename)
    new_filename = f"{basename}_{suffix.lower()}{ext}"
    new_file_path = os.path.join(dirname, new_filename)
    
    # Read the content of the original file
    with open(original_pbtxt, 'r') as file:
        content = file.read()
    
    # Replace calculator names with their suffixed versions
    calculator_names = [calc['name'] for calc in calculators if calc.get('name')]
    for name in calculator_names:
        content = re.sub(
            rf'calculator:\s*"{name}"',
            f'calculator: "{name}{suffix}"',
            content
        )
    
    # Write the modified content to the new file
    with open(new_file_path, 'w') as file:
        file.write(content)
    
    print(f"Created {new_file_path}")
    return new_file_path


def main():
    """Main function to run the calculator cloning script."""
    parser = argparse.ArgumentParser(description='Clone MediaPipe calculators with a specified suffix.')
    parser.add_argument('--pipeline_json', required=True, 
                        help='Path to the JSON file describing the pipeline hierarchy')
    parser.add_argument('--pbtxt_file', required=True,
                        help='Path to the pbtxt file to clone')
    parser.add_argument('--suffix', required=True,
                        help='Suffix to add to calculator names')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    pipeline_json = os.path.abspath(args.pipeline_json)
    pbtxt_file = os.path.abspath(args.pbtxt_file)
    suffix = args.suffix
    
    # Read the JSON file
    json_data = read_json_file(pipeline_json)
    
    # Extract calculators from the JSON data
    calculators = extract_calculators_from_json(json_data)
    print(f"Found {len(calculators)} calculators in the pipeline")
    
    # Clone each calculator file
    for calc in calculators:
        if calc.get('source'):
            clone_calculator_file(calc['source'], suffix)
    
    # Update BUILD files
    update_build_file(calculators, suffix)
    update_hand_tracking_build(suffix)
    
    # Clone the pbtxt file
    clone_pbtxt_file(pbtxt_file, suffix, calculators)
    
    print(f"Successfully cloned {len(calculators)} calculators with suffix '{suffix}'")


if __name__ == "__main__":
    main()
