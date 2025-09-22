#!/usr/bin/env python3
"""
MediaPipe Pipeline Parser

Parses MediaPipe .pbtxt graph definition files to extract the computational
graph structure and identify all calculator nodes and their connections.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import yaml


@dataclass
class GraphNode:
    """Represents a calculator node in the MediaPipe graph."""
    name: str
    input_streams: List[str]
    output_streams: List[str]
    input_side_packets: List[str]
    output_side_packets: List[str]
    node_options: Dict[str, str]
    line_number: int

@dataclass
class PipelineNode:
    name: str
    node_type: str  # 'calculator', 'subgraph', or 'other'
    source: Optional[str]
    description: Optional[str]
    children: List[Any]  # List[PipelineNode], but recursive types need Any

@dataclass
class MediaPipeGraph:
    """Represents the complete MediaPipe computation graph."""
    input_streams: List[str]
    output_streams: List[str]
    input_side_packets: List[str]
    output_side_packets: List[str]
    nodes: List[GraphNode]
    packet_generators: List[Dict[str, str]]


class MediaPipePipelineParser:

    def print_with_ident(self, text: str):
        print(4 * ' ' * self.ident + text)

    def __init__(self, mediapipe_source_path: Path):

        self.mediapipe_source = Path(mediapipe_source_path)

        # map all calculator and graph sources across the mediapipe source code
        self.calculators_source_mapping, self.graph_source_files = self.map_node_sources()

        self.ident = 0

        
    def parse_pbtxt_file(self, pbtxt_path: Path) -> MediaPipeGraph:
        """
        Parse a .pbtxt file and extract the graph structure.
        
        Args:
            pbtxt_path: Path to the .pbtxt file
            
        Returns:
            MediaPipeGraph object representing the parsed graph
        """

        with open(pbtxt_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        graph = MediaPipeGraph(
            input_streams=[],
            output_streams=[],
            input_side_packets=[],
            output_side_packets=[],
            nodes=[],
            packet_generators=[]
        )
        
        # Parse top-level streams and packets
        graph.input_streams = self._extract_pbtxt_field(content, "input_stream")
        graph.output_streams = self._extract_pbtxt_field(content, "output_stream")
        graph.input_side_packets = self._extract_pbtxt_field(content, "input_side_packet")
        graph.output_side_packets = self._extract_pbtxt_field(content, "output_side_packet")
        
        # Parse nodes
        graph.nodes = self._parse_nodes(content, lines)
        
        # Parse packet generators
        graph.packet_generators = self._parse_packet_generators(content)
        
        return graph
    
    def _extract_pbtxt_field(self, content: str, field_name: str) -> List[str]:
        """Extract list field values from pbtxt content."""
        pattern = rf'^{field_name}\s*:\s*"([^"]*)"'
        matches = re.findall(pattern, content)
        return matches
    
    def _parse_nodes(self, content: str, lines: List[str]) -> List[GraphNode]:
        """Parse calculator nodes from the graph."""
        nodes = []
        
        # Find all node blocks
        node_pattern = r'node\s*\{'
        node_starts = []
        for i, line in enumerate(lines):
            if re.search(node_pattern, line):
                node_starts.append(i)
        
        for start_line in node_starts:
            node = self._parse_single_node(lines, start_line)
            if node:
                nodes.append(node)
        
        return nodes
    
    def _parse_single_node(self, lines: List[str], start_line: int) -> Optional[GraphNode]:
        """Parse a single node block."""
        # Find the end of this node block
        brace_count = 0
        end_line = start_line
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > start_line:
                end_line = i
                break
        
        node_content = '\n'.join(lines[start_line:end_line+1])
        
        # Extract node fields
        name = self._extract_field(node_content, "calculator")
        
        if not name:
            return None
        
        input_streams = self._extract_pbtxt_field(node_content, "input_stream")
        output_streams = self._extract_pbtxt_field(node_content, "output_stream")
        input_side_packets = self._extract_pbtxt_field(node_content, "input_side_packet")
        output_side_packets = self._extract_pbtxt_field(node_content, "output_side_packet")
        
        # Extract node options (simplified)
        node_options = {}
        options_match = re.search(r'node_options\s*\{([^}]*)\}', node_content, re.DOTALL)
        if options_match:
            options_content = options_match.group(1)
            # This is a simplified parser - could be expanded for complex options
            for line in options_content.split('\n'):
                if ':' in line:
                    key_value = line.strip().split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip().strip('"')
                        node_options[key] = value
        
        return GraphNode(
            name=name or "",
            input_streams=input_streams,
            output_streams=output_streams,
            input_side_packets=input_side_packets,
            output_side_packets=output_side_packets,
            node_options=node_options,
            line_number=start_line + 1
        )

    def _parse_packet_generators(self, content: str) -> List[Dict[str, str]]:
        """Parse packet generator definitions."""
        generators = []
        
        generator_pattern = r'packet_generator\s*\{([^}]*)\}'
        matches = re.findall(generator_pattern, content, re.DOTALL)
        
        for match in matches:
            generator = {}
            for line in match.split('\n'):
                if ':' in line:
                    key_value = line.strip().split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip().strip('"')
                        generator[key] = value
            if generator:
                generators.append(generator)
        
        return generators
    
    def _extract_field(self, content: str, field_name: str) -> Optional[str]:
        """Extract a single field value from pbtxt content."""
        pattern = rf'{field_name}\s*:\s*"([^"]*)"'
        match = re.search(pattern, content)
        return match.group(1) if match else None
    
    def map_node_sources(self) -> tuple[dict[str, Path], dict[str, Path]]:

        """ Scan the entire mediapipe source tree to map node names to the source files defining them.

          this is accomplished for two types of nodes:
            + nodes which are calculators
            + nodes which are graphs

         the scan compiles this mapping for all such nodes, so that the caller
         is sure to find the source file defining any node by its name through
         this returned mapping. """

        cpp_suffixes = [
            # Source files
            ".cc", ".cpp", ".cxx", ".c++", ".C", ".cp", ".CPP",
            # Header files
            ".h", ".hpp", ".hh", ".hxx", ".H",
            # Other includable fragments
            ".inc", ".inl"
            ]

        calculator_source_files = dict[str, Path]()
        graph_source_files = dict[str, Path]()

        # scan the source tree to associate node names to their source files definition locations
        for root, dirs, files in self.mediapipe_source.walk():
            for file in files:

                source_file = root / file
                file_suffix = Path(file).suffix

                # associate calculator and pipeline nodes defined in c++ source files
                if file_suffix in cpp_suffixes:

                    source_code = source_file.read_text()
                    # associate all calculators defined in this source file by the REGISTER_CALCULATOR macro.
                    # (the REGISTER_CALCULATOR itself is defined in mediapipe/framework/calculator_registry.h.
                    # a code comment mentions that MEDIAPIPE_REGISTER_FACTORY_FUNCTION_QUALIFIED should be migrated to
                    # so if calculators are not all found by the current function explore scanning the mediapipe source
                    # code for also MEDIAPIPE_REGISTER_FACTORY_FUNCTION_QUALIFIED.
                    for calculator_name in re.findall(r'REGISTER_CALCULATOR\((.*)\)', source_code):
                        if str(source_file.with_suffix('')).endswith('test'):
                            pass
                        else:
                            if calculator_name in calculator_source_files:
                                calculator_source_files[calculator_name] = [calculator_source_files[calculator_name]] + [source_file]
                            else:
                                calculator_source_files[calculator_name] = source_file

                    # associate all calculators defined by the MEDIAPIPE_REGISTER_NODE macro, whose own comments (search define MEDIAPIPE_REGISTER_NODE)
                    # say is deprecated, but still is used for defining a few calculators.
                    for calculator_name in re.findall(r'MEDIAPIPE_REGISTER_NODE\((.*)\)', source_code):
                        if str(source_file.with_suffix('')).endswith('test'):
                            pass
                        else:
                            if calculator_name in calculator_source_files:
                                calculator_source_files[calculator_name] = [calculator_source_files[calculator_name]] + [source_file]
                            else:
                                calculator_source_files[calculator_name] = source_file

                    # associate pipelines nodes defined in .cc files
                    for source_graph_name in re.findall(r'REGISTER_MEDIAPIPE_GRAPH\((.*)\)', source_code):
                        if str(source_file.with_suffix('')).endswith('test'):
                            pass
                        else:
                            if source_graph_name in graph_source_files:
                                graph_source_files[source_graph_name] = [graph_source_files[source_graph_name]] + [source_file]
                            else:
                                graph_source_files[source_graph_name] = source_file

                # associate pipeline nodes defined in .pbtxt files
                if file_suffix == '.pbtxt':

                    graph_source_file = root / file
                    graph_source_code = graph_source_file.read_text()

                    matches = re.findall(r'type\s*:\s*"([^"]+)"', graph_source_code)
                    if len(matches) == 1:
                        graph_name = matches[0]
                        graph_source_files[graph_name] = graph_source_file

        # redact any calculator name which had more than one source file found for it
        for calculator_name in list(calculator_source_files.keys()):
            if isinstance(calculator_source_files[calculator_name], list):
                print(f'🤷 the calculator name \'{calculator_name}\' is defined in more than one source file and will be ignored; should switch to yielding calculator source files by fully qualified module paths, or filter the source directories being searched. '
                      f'this name is currently defined as a calculator in the following source files:')
                for source_file in calculator_source_files[calculator_name]:
                    print(f'  - {source_file}')
                del calculator_source_files[calculator_name]

        print(f'🛈 mapped {len(calculator_source_files)} calculator names to the source file defining them')
        print(f'🛈 mapped {len(graph_source_files)} graph definitions to the source file defining them')


        # sanity review the the resulting associations from names to source files
        # for calculator_name, paths in calculator_source_files.items():
        #     if len(paths) == 0:
        #         raise Exception(f'internal error')
        #     elif len(paths) == 1:
        #         print(f'⚠️️️ calculator \'{calculator_name}\' was associated with a single source file and not a .cc and .h pair: {paths[0]}')
        #     elif len(paths) == 2:
        #         if paths[0].with_suffix('') == paths[1].with_suffix(''):
        #             continue
        #         else:
        #             print(f'⚠️ calculator \'{calculator_name}\' was associated with two source files which are not a .cc and .h pair:')
        #             for path in paths:
        #                 print(f'  - {path}')
        #     else:
        #         print(f'⚠️ calculator \'{calculator_name}\' was associated with a plurality of source files:')
        #         for path in paths:
        #             print(f'  - {path}')
        
        return calculator_source_files, graph_source_files
    
    # def find_graph_definitions(self) -> Dict[str, Path]:
    #     """Find all graph definition .pbtxt files."""
    #     graph_files = {}
    #
    #     for root, dirs, files in os.walk(self.mediapipe_source):
    #         for file in files:
    #             if file.endswith('.pbtxt'):
    #                 path = Path(root) / file
    #                 graph_name = file.replace('.pbtxt', '')
    #                 graph_files[graph_name] = path
    #
    #     return graph_files
    
    def _camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.capitalize() for x in components)

    def analyze_pipeline(self, pipeline_name: str, parent_output_dir: Optional[Path] = None) -> PipelineNode:
        """Analyze the specific hand landmark tracking pipeline and build a hierarchy tree."""
        if pipeline_name in self.graph_source_files:
            self.print_with_ident(f'analyzing the pipeline definition of \'{pipeline_name}\' found at {self.graph_source_files[pipeline_name]}')
        else:
            raise FileNotFoundError(f'could not find a pipeline graph definition of {pipeline_name} in mediapipe source code')

        # parse the given graph
        if str(self.graph_source_files[pipeline_name]).endswith('.pbtxt'):
            graph = self.parse_pbtxt_file(self.graph_source_files[pipeline_name])
        else:
            self.print_with_ident(f'⚠️ pipeline {pipeline_name} is defined by C++ code and not a .pbtxt file; its content will be ignored from analysis.')
            return PipelineNode(
                name=pipeline_name,
                node_type='subgraph',
                source=str(self.graph_source_files[pipeline_name]),
                description='Defined in C++ code, not a .pbtxt file.',
                children=[]
            )

        calculator_mapping = dict[str, Path]()
        subgraph_mapping = dict[str, Path]()
        children = []

        for node in graph.nodes:
            node_name = node.name
            node_source = None
            node_type = 'other'
            description = None
            sub_children = []

            if node_name in self.calculators_source_mapping:
                node_source = str(self.calculators_source_mapping[node_name])
                calculator_mapping[node_name] = node_source
                node_type = 'calculator'
                self.print_with_ident(f'✅ calculator node \'{node_name}\' is defined in source file {node_source}')

            if node_name in self.graph_source_files:
                if node_source:
                    self.print_with_ident(f'❌️️️ node \'{node_name}\' is found to be both a calculator and a sub-graph; this case is not currently handled downstream!')
                node_source = str(self.graph_source_files[node_name])
                subgraph_mapping[node_name] = node_source
                node_type = 'subgraph'
                self.print_with_ident(f'✅ sub-graph node \'{node_name}\' is defined in source file {node_source}')
                self.print_with_ident(f'🔁 \'{node_name}\' is a sub-graph of the current one')
                self.ident += 1
                sub_children = [self.analyze_pipeline(node_name, parent_output_dir)]
                self.ident -= 1

            if not node_source:
                match node_name:
                    case 'InferenceCalculator':
                        description = (
                            f'⚠️ node \'{node_name}\' is a node name not associated to one implementation, but a generic name which has multiple implementions in mediapipe/calculators/tensor/inference_calculator.h '
                            f'one per each target inference stack (e.g. cpu, xnnpack, and more) which subclass it in that source file. assume the one selected at runtime matches the target inference type used '
                            f'by the pipeline being run. all implementations of it run a neural network ℹ️ see also https://ai.google.dev/edge/mediapipe/framework/framework_concepts/graphs_cpp')
                    case 'LandmarkProjectionCalculator':
                        description = (
                            f'⚠️ node \'{node_name}\' is a node name not associated to one implementation, but a generic name which has an implemention per pipeline each '
                            f'defined by the form graph.AddNode("LandmarkProjectionCalculator"), its content will be ignored from analysis, but you can manually locate '
                            f'all specific implementions of it by searching the occurences of graph.AddNode("LandmarkProjectionCalculator") in the mediapipe source tree.'
                            f'AddNode is defined in builder.h and elsewhere and dynamically instantiates a LandmarkProjectionCalculator instance based on context '
                            f'ℹ️ see also https://ai.google.dev/edge/mediapipe/framework/framework_concepts/graphs')
                    case 'WorldLandmarkProjectionCalculator':
                        description = (
                            f'⚠️ node \'{node_name}\' is a node name not associated to one implementation, but a generic name which has an implemention per pipeline each '
                            f'defined by the form graph.AddNode("WorldLandmarkProjectionCalculator"), its content will be ignored from analysis, but you can manually locate '
                            f'all specific implementions of it by searching the occurences of graph.AddNode("WorldLandmarkProjectionCalculator") in the mediapipe source tree. '
                            f'AddNode is defined in builder.h and elsewhere and dynamically instantiates a WorldLandmarkProjectionCalculator instance based on context '
                            f'ℹ️ see also https://ai.google.dev/edge/mediapipe/framework/framework_concepts/graphs')
                    case _:
                        description = f'❌ could not locate the source code for node \'{node_name}\''
                node_type = 'other'

            children.append(PipelineNode(
                name=node_name,
                node_type=node_type,
                source=node_source,
                description=description,
                children=sub_children
            ))

        return PipelineNode(
            name=pipeline_name,
            node_type='subgraph',
            source=str(self.graph_source_files[pipeline_name]),
            description=None,
            children=children
        )

    def write_pipeline_outputs(self, root_node: PipelineNode, output_dir: Path):
        """Write the pipeline hierarchy to markdown, JSON, and YAML files in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write JSON
        json_path = output_dir / 'pipeline.json'
        with open(json_path, 'w') as f:
            json.dump(self._node_to_dict_rel(root_node), f, indent=2)
        # Write YAML
        yaml_path = output_dir / 'pipeline.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(self._node_to_dict_rel(root_node), f, sort_keys=False, allow_unicode=True)
        # Write Markdown
        md_path = output_dir / 'pipeline.md'
        with open(md_path, 'w') as f:
            f.write(self._node_to_markdown(root_node, 0, json_path, yaml_path))

    def _node_to_dict_rel(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary, making 'source' fields relative to cwd."""
        def rel(p):
            if p is None:
                return None
            try:
                return str(Path(p).relative_to(Path.cwd()))
            except Exception:
                return str(p)
        return {
            'name': node.name,
            'type': node.node_type,
            'source': rel(node.source),
            'description': node.description,
            'children': [self._node_to_dict_rel(child) for child in node.children]
        }

    def _node_to_markdown(self, node: PipelineNode, level: int, json_path: Path, yaml_path: Path) -> str:
        indent = '    ' * level
        if level == 0:
            # Add links to JSON and YAML at the top, relative to the markdown file location
            md = f'# Pipeline Hierarchy for `{node.name}`\n\n'
            md += f'[JSON version]({json_path.name}) | [YAML version]({yaml_path.name})\n\n'
        else:
            md = ''
        # Node name as link or description
        if node.source:
            md += f'{indent}- [{node.name}]({Path(node.source).resolve()})'
        elif node.description:
            md += f'{indent}- **{node.name}**: {node.description}'
        else:
            md += f'{indent}- **{node.name}**'
        md += '\n'
        for child in node.children:
            md += self._node_to_markdown(child, level + 1, json_path, yaml_path)
        return md

    def _node_to_dict(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary for serialization."""
        return {
            'name': node.name,
            'type': node.node_type,
            'source': node.source,
            'description': node.description,
            'children': [self._node_to_dict(child) for child in node.children]
        }

def main():
    """Test the pipeline parser and write outputs."""
    import sys
    parser = MediaPipePipelineParser(Path('mediapipe').resolve())
    root_node = parser.analyze_pipeline('HandLandmarkTrackingCpu')
    # Output directory is always mediapipe_analysis/analysis/output relative to cwd
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path('mediapipe_analysis/analysis/output')
    parser.write_pipeline_outputs(root_node, output_dir)
    # Print file:// links to the output files (absolute paths)
    json_path = (output_dir / 'pipeline.json').resolve()
    yaml_path = (output_dir / 'pipeline.yaml').resolve()
    md_path = (output_dir / 'pipeline.md').resolve()
    print(f"\nPipeline outputs written:")
    print(f"  Markdown: file://{md_path}")
    print(f"  JSON:     file://{json_path}")
    print(f"  YAML:     file://{yaml_path}")

if __name__ == "__main__":
    main()