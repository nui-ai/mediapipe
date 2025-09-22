"""
MediaPipe Pipeline Parser

Parses MediaPipe .pbtxt graph definition files to extract the computational
graph structure and identify all calculator nodes and their connections.
"""

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import yaml



@dataclass
class PipelineNode:
    name: str
    node_type: str  # 'calculator' or 'subgraph'
    source: Optional[str]
    description: Optional[str]
    children: List['PipelineNode']
    source_line_number: Optional[int] = None
    source_line_code: Optional[str] = None
    input_streams: Optional[list] = None
    output_streams: Optional[list] = None
    input_side_packets: Optional[list] = None
    output_side_packets: Optional[list] = None
    node_options: Optional[dict] = None

@dataclass
class MediaPipeGraph:
    """Represents the complete MediaPipe computation graph."""
    input_streams: List[str]
    output_streams: List[str]
    input_side_packets: List[str]
    output_side_packets: List[str]
    nodes: list  # List of node dicts or PipelineNode, not GraphNode
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
        matches = re.findall(pattern, content, re.MULTILINE)
        return matches
    
    def _parse_nodes(self, content: str, lines: List[str]) -> list:
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
    
    def _parse_single_node(self, lines: List[str], start_line: int) -> Optional[dict]:
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
        # Enhanced node_options parsing (handles nested braces, simple key-value pairs)
        node_options = {}
        options_match = re.search(r'node_options\s*\{([^}]*)\}', node_content, re.DOTALL)
        if options_match:
            options_content = options_match.group(1)
            brace_level = 0
            current_key = None
            for line in options_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if '{' in line:
                    brace_level += 1
                if '}' in line:
                    brace_level -= 1
                if ':' in line and brace_level == 0:
                    key_value = line.split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip().strip('"')
                        node_options[key] = value
        return {
            'name': name or "",
            'input_streams': input_streams,
            'output_streams': output_streams,
            'input_side_packets': input_side_packets,
            'output_side_packets': output_side_packets,
            'node_options': node_options,
            'line_number': start_line + 1
        }

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

        calculator_source_files = dict[str, tuple[Path, int, str]]()
        graph_source_files = dict[str, tuple[Path, int, str]]()

        # scan the source tree to associate node names to their source files definition locations
        # For each file, we look for macro registrations and record the file, line number, and code line for each match.
        for root, dirs, files in self.mediapipe_source.walk():
            for file in files:
                source_file = root / file
                file_suffix = Path(file).suffix
                # For C++ source/header files, look for registration macros
                if file_suffix in cpp_suffixes:
                    lines = source_file.read_text().splitlines()
                    # REGISTER_CALCULATOR: Associates calculators defined by the macro to their source file and line
                    for i, line in enumerate(lines):
                        m = re.match(r'.*REGISTER_CALCULATOR\(([^)]+)\)', line)
                        if m:
                            calculator_name = m.group(1).strip()
                            if str(source_file.with_suffix('')).endswith('test'):
                                continue
                            calculator_source_files[calculator_name] = (source_file, i+1, line.strip())
                    # MEDIAPIPE_REGISTER_NODE: Associates calculators defined by the deprecated macro to their source file and line
                    for i, line in enumerate(lines):
                        m = re.match(r'.*MEDIAPIPE_REGISTER_NODE\(([^)]+)\)', line)
                        if m:
                            calculator_name = m.group(1).strip()
                            if str(source_file.with_suffix('')).endswith('test'):
                                continue
                            calculator_source_files[calculator_name] = (source_file, i+1, line.strip())
                    # REGISTER_MEDIAPIPE_GRAPH: Associates graph nodes defined by the macro to their source file and line
                    for i, line in enumerate(lines):
                        m = re.match(r'.*REGISTER_MEDIAPIPE_GRAPH\(([^)]+)\)', line)
                        if m:
                            graph_name = m.group(1).strip()
                            if str(source_file.with_suffix('')).endswith('test'):
                                continue
                            graph_source_files[graph_name] = (source_file, i+1, line.strip())
                # For .pbtxt files, associate graph nodes by type (line navigation is not relevant to them)
                if file_suffix == '.pbtxt':
                    graph_source_file = root / file
                    graph_source_code = graph_source_file.read_text()
                    matches = re.findall(r'type\s*:\s*"([^"]+)"', graph_source_code)
                    if len(matches) == 1:
                        graph_name = matches[0]
                        graph_source_files[graph_name] = (graph_source_file, 1, None)

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
            src_info = self.graph_source_files[pipeline_name]
            graph_path = str(src_info[0])
            graph_line_number = src_info[1]
            graph_line_code = src_info[2]
            self.print_with_ident(f'analyzing the pipeline definition of \'{pipeline_name}\' found at {graph_path}')
        else:
            raise FileNotFoundError(f'could not find a pipeline graph definition of {pipeline_name} in mediapipe source code')
        # parse the given graph
        if graph_path.endswith('.pbtxt'):
            graph = self.parse_pbtxt_file(graph_path)
        else:
            self.print_with_ident(f'⚠️ pipeline {pipeline_name} is defined by C++ code and not a .pbtxt file; its content will be ignored from analysis.')
            return PipelineNode(
                name=pipeline_name,
                node_type='subgraph',
                source=graph_path,
                description='Defined in C++ code, not a .pbtxt file.',
                children=[],
                source_line_number=graph_line_number,
                source_line_code=graph_line_code
            )

        calculator_mapping = dict[str, Path]()
        subgraph_mapping = dict[str, Path]()
        children = []
        for node in graph.nodes:
            # Extract stream/packet/options fields from the node (which is a dict)
            node_name = getattr(node, 'name', None) or node.get('name')
            input_streams = getattr(node, 'input_streams', None) or node.get('input_streams')
            output_streams = getattr(node, 'output_streams', None) or node.get('output_streams')
            input_side_packets = getattr(node, 'input_side_packets', None) or node.get('input_side_packets')
            output_side_packets = getattr(node, 'output_side_packets', None) or node.get('output_side_packets')
            node_options = getattr(node, 'node_options', None) or node.get('node_options')
            node_type = 'other'
            description = None
            sub_children = []
            source_line_number = None
            source_line_code = None
            node_source = None  # Always initialize
            is_calc = node_name in self.calculators_source_mapping
            is_subgraph = node_name in self.graph_source_files
            if is_calc and is_subgraph:
                self.print_with_ident(f'❌️️️ node \'{node_name}\' is found to be both a calculator and a sub-graph; this case is not currently handled downstream!')
            if is_calc:
                src_info = self.calculators_source_mapping[node_name]
                node_source = str(src_info[0])
                source_line_number = src_info[1]
                source_line_code = src_info[2]
                calculator_mapping[node_name] = node_source
                node_type = 'calculator'
                self.print_with_ident(f'✅ calculator node \'{node_name}\' is defined in source file {node_source}')
            if is_subgraph:
                src_info = self.graph_source_files[node_name]
                node_source = str(src_info[0])
                source_line_number = src_info[1]
                source_line_code = src_info[2]
                subgraph_mapping[node_name] = node_source
                node_type = 'subgraph'
                self.print_with_ident(f'✅ sub-graph node \'{node_name}\' is defined in source file {node_source}')
                self.print_with_ident(f'🔁 \'{node_name}\' is a sub-graph of the current one')
                self.ident += 1
                sub_children = [self.analyze_pipeline(node_name, parent_output_dir)]
                self.ident -= 1
            if not (is_calc or is_subgraph):
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
                children=sub_children,
                source_line_number=source_line_number,
                source_line_code=source_line_code,
                input_streams=input_streams,
                output_streams=output_streams,
                input_side_packets=input_side_packets,
                output_side_packets=output_side_packets,
                node_options=node_options
            ))
        # Set top-level streams/packets on the root node
        return PipelineNode(
            name=pipeline_name,
            node_type='subgraph',
            source=graph_path,
            description=None,
            children=children,
            source_line_number=graph_line_number,
            source_line_code=graph_line_code,
            input_streams=graph.input_streams,
            output_streams=graph.output_streams,
            input_side_packets=graph.input_side_packets,
            output_side_packets=graph.output_side_packets,
            node_options=None
        )

    def write_pipeline_outputs(self, root_node: PipelineNode, output_dir: Path):
        """Write the pipeline hierarchy to markdown (with and without line numbers), JSON, and YAML files in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write JSON (basic)
        json_basic_path = output_dir / 'pipeline.basic.json'
        with open(json_basic_path, 'w') as f:
            json.dump(self._node_to_dict_rel(root_node), f, indent=2)
        # Write YAML (basic)
        yaml_basic_path = output_dir / 'pipeline.basic.yaml'
        with open(yaml_basic_path, 'w') as f:
            yaml.dump(self._node_to_dict_rel(root_node), f, sort_keys=False, allow_unicode=True)
        # Write Markdown (basic, with line numbers)
        md_basic_path = output_dir / 'pipeline.basic.md'
        with open(md_basic_path, 'w') as f:
            f.write(self._node_to_markdown_with_links(root_node, 0, json_basic_path, yaml_basic_path, True))
        # Write Markdown (basic, no line numbers)
        md_basic_noline_path = output_dir / 'pipeline.basic.noline.md'
        with open(md_basic_noline_path, 'w') as f:
            f.write(self._node_to_markdown_with_links(root_node, 0, json_basic_path, yaml_basic_path, False))
        # Write JSON (verbose)
        json_verbose_path = output_dir / 'pipeline.verbose.json'
        with open(json_verbose_path, 'w') as f:
            json.dump(self._node_to_dict_verbose(root_node), f, indent=2)
        # Write YAML (verbose)
        yaml_verbose_path = output_dir / 'pipeline.verbose.yaml'
        with open(yaml_verbose_path, 'w') as f:
            yaml.dump(self._node_to_dict_verbose(root_node), f, sort_keys=False, allow_unicode=True)
        # Write Markdown (verbose, with line numbers)
        md_verbose_path = output_dir / 'pipeline.verbose.md'
        with open(md_verbose_path, 'w') as f:
            f.write(self._node_to_markdown_with_links_verbose(root_node, 0, json_verbose_path, yaml_verbose_path, md_basic_path, md_basic_noline_path, json_basic_path, yaml_basic_path, True, md_verbose_path, yaml_verbose_path, json_verbose_path))

    def _node_to_dict_rel(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary, making 'source' fields relative to cwd, and using 'nodes' instead of 'children'."""
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
            'nodes': [self._node_to_dict_rel(child) for child in node.children]
        }

    def _node_to_dict(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary for serialization, using 'nodes' instead of 'children'."""
        return {
            'name': node.name,
            'type': node.node_type,
            'source': node.source,
            'description': node.description,
            'nodes': [self._node_to_dict(child) for child in node.children]
        }

    def _node_to_dict_verbose(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary for verbose serialization (all fields), using 'nodes' instead of 'children'."""
        return {
            'name': node.name,
            'type': node.node_type,
            'source': node.source,
            'description': node.description,
            'source_line_number': node.source_line_number,
            'input_streams': node.input_streams,
            'output_streams': node.output_streams,
            'input_side_packets': node.input_side_packets,
            'output_side_packets': node.output_side_packets,
            'node_options': node.node_options,
            'nodes': [self._node_to_dict_verbose(child) for child in node.children]
        }

    def _node_to_markdown_with_links(self, node: PipelineNode, level: int, json_path: Path, yaml_path: Path, with_line_numbers: bool) -> str:
        indent = '    ' * level
        if level == 0:
            abs_dir = Path(json_path).parent.resolve()
            md = f'# Pipeline Hierarchy for `{node.name}`\n\n'
            md += '| Format | Description | Link |\n'
            md += '|--------|-------------|------|\n'
            md += f'| Basic Markdown (with line numbers) | Tree with hyperlinks to each node\'s source (with line numbers) | [pipeline.basic.md](file://{abs_dir}/pipeline.basic.md) |\n'
            md += f'| Basic Markdown (no line numbers) | Tree with hyperlinks to each node\'s source (no line numbers) | [pipeline.basic.noline.md](file://{abs_dir}/pipeline.basic.noline.md) |\n'
            md += f'| Basic JSON | Tree in JSON format | [pipeline.basic.json](file://{abs_dir}/pipeline.basic.json) |\n'
            md += f'| Basic YAML | Tree in YAML format | [pipeline.basic.yaml](file://{abs_dir}/pipeline.basic.yaml) |\n'
            md += f'| Verbose Markdown | Tree with more node fields (streams, packets, options) | [pipeline.verbose.md](file://{abs_dir}/pipeline.verbose.md) |\n'
            md += f'| Verbose JSON | Tree with more node fields (streams, packets, options) | [pipeline.verbose.json](file://{abs_dir}/pipeline.verbose.json) |\n'
            md += f'| Verbose YAML | Tree with more node fields (streams, packets, options) | [pipeline.verbose.yaml](file://{abs_dir}/pipeline.verbose.yaml) |\n\n'
        else:
            md = ''
        # Node name as link or description
        if node.node_type == 'calculator' and node.source and node.source_line_number:
            if with_line_numbers:
                link = f"{Path(node.source).resolve()}#L{node.source_line_number}"
                md += f'{indent}- [{node.name}]({link})'
            else:
                md += f'{indent}- [{node.name}]({Path(node.source).resolve()})'
        elif node.source:
            md += f'{indent}- [{node.name}]({Path(node.source).resolve()})'
        elif node.description:
            md += f'{indent}- **{node.name}**: {node.description}'
        else:
            md += f'{indent}- **{node.name}**'
        md += '\n'
        for child in node.children:
            md += self._node_to_markdown_with_links(child, level + 1, json_path, yaml_path, with_line_numbers)
        return md

    def _node_to_markdown_with_links_verbose(self, node: PipelineNode, level: int, json_verbose_path: Path, yaml_verbose_path: Path, md_basic_path: Path, md_basic_noline_path: Path, json_basic_path: Path, yaml_basic_path: Path, with_line_numbers: bool, md_verbose_path: Path, yaml_verbose_path2: Path, json_verbose_path2: Path) -> str:
        indent = '    ' * level
        if level == 0:
            abs_dir = Path(json_verbose_path).parent.resolve()
            md = f'# Pipeline Hierarchy for `{node.name}` (verbose)\n\n'
            md += '| Format | Description | Link |\n'
            md += '|--------|-------------|------|\n'
            md += f'| Basic Markdown (with line numbers) | Tree with hyperlinks to each node\'s source (with line numbers) | [pipeline.basic.md](file://{abs_dir}/pipeline.basic.md) |\n'
            md += f'| Basic Markdown (no line numbers) | Tree with hyperlinks to each node\'s source (no line numbers) | [pipeline.basic.noline.md](file://{abs_dir}/pipeline.basic.noline.md) |\n'
            md += f'| Basic JSON | Tree in JSON format | [pipeline.basic.json](file://{abs_dir}/pipeline.basic.json) |\n'
            md += f'| Basic YAML | Tree in YAML format | [pipeline.basic.yaml](file://{abs_dir}/pipeline.basic.yaml) |\n'
            md += f'| Verbose Markdown | Tree with all node fields (streams, packets, options) | [pipeline.verbose.md](file://{abs_dir}/pipeline.verbose.md) |\n'
            md += f'| Verbose JSON | Tree with all node fields (streams, packets, options) | [pipeline.verbose.json](file://{abs_dir}/pipeline.verbose.json) |\n'
            md += f'| Verbose YAML | Tree with all node fields (streams, packets, options) | [pipeline.verbose.yaml](file://{abs_dir}/pipeline.verbose.yaml) |\n\n'
        else:
            md = ''
        # Node name as link or description
        if node.node_type == 'calculator' and node.source and node.source_line_number:
            if with_line_numbers:
                link = f"{Path(node.source).resolve()}#L{node.source_line_number}"
                md += f'{indent}- [{node.name}]({link})'
            else:
                md += f'{indent}- [{node.name}]({Path(node.source).resolve()})'
        elif node.source:
            md += f'{indent}- [{node.name}]({Path(node.source).resolve()})'
        elif node.description:
            md += f'{indent}- **{node.name}**: {node.description}'
        else:
            md += f'{indent}- **{node.name}**'
        md += f'\n{indent}  - **input_streams:** {node.input_streams if node.input_streams is not None else []}'
        md += f'\n{indent}  - **output_streams:** {node.output_streams if node.output_streams is not None else []}'
        md += f'\n{indent}  - **input_side_packets:** {node.input_side_packets if node.input_side_packets is not None else []}'
        md += f'\n{indent}  - **output_side_packets:** {node.output_side_packets if node.output_side_packets is not None else []}'
        md += f'\n{indent}  - **node_options:** {json.dumps(node.node_options, indent=2) if node.node_options else {}}'
        md += '\n'
        for child in node.children:
            md += self._node_to_markdown_with_links_verbose(child, level + 1, json_verbose_path, yaml_verbose_path, md_basic_path, md_basic_noline_path, json_basic_path, yaml_basic_path, with_line_numbers, md_verbose_path, yaml_verbose_path2, json_verbose_path2)
        return md

def main():
    """ run the pipeline parser  """
    parser = MediaPipePipelineParser(Path('mediapipe').resolve())
    print()
    root_node = parser.analyze_pipeline('HandLandmarkTrackingCpu')
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path('mediapipe_analysis/analysis/output')
    parser.write_pipeline_outputs(root_node, output_dir)
    output_dir_abs = output_dir.resolve()
    print(f"\nPipeline parsing outputs are available in:\nfile://{output_dir_abs}/\n")
    print("Markdown with hyperlinks to each node's source ― without line numbers for source hyperlinks, as they are not a standard markdown feature:")
    print(f"file://{output_dir_abs}/pipeline.basic.noline.md")
    print("Markdown with hyperlinks to each node's source ― source hyperlinks include line numbers:")
    print(f"file://{output_dir_abs}/pipeline.basic.md")
    print("Markdown (verbose): includes more graph node fields: input/output streams, side packets, and node options:")
    print(f"file://{output_dir_abs}/pipeline.verbose.md")
    print("\nJSON:")
    print(f"file://{output_dir_abs}/pipeline.basic.json")
    print("JSON (verbose): includes more graph node fields: input/output streams, side packets, and node options:")
    print(f"file://{output_dir_abs}/pipeline.verbose.json")
    print("\nYAML:")
    print(f"file://{output_dir_abs}/pipeline.basic.yaml")
    print("YAML (verbose): includes more graph node fields: input/output streams, side packets, and node options:")
    print(f"file://{output_dir_abs}/pipeline.verbose.yaml")

if __name__ == "__main__":
    main()
