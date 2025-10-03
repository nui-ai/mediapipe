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
import os

@dataclass
class GraphSelfDescription:
    input_streams: Optional[list] = None
    output_streams: Optional[list] = None
    input_side_packets: Optional[list] = None
    output_side_packets: Optional[list] = None
    node_options: Optional[dict] = None
    description: Optional[str] = None

@dataclass
class PipelineNode:
    name: str
    node_type: str  # 'calculator' or 'graph'
    source: Optional[str]
    warning: Optional[str]
    children: List['PipelineNode']
    source_line_number: Optional[int] = None
    source_line_code: Optional[str] = None
    input_streams: Optional[list] = None
    output_streams: Optional[list] = None
    input_side_packets: Optional[list] = None
    output_side_packets: Optional[list] = None
    node_options: Optional[dict] = None
    description: Optional[str] = None
    graph_self_description: Optional[GraphSelfDescription] = None

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

        
    def parse_pbtxt_file(self, pbtxt_path: Path) -> tuple[MediaPipeGraph, Optional[str]]:
        """
        Parse a .pbtxt file and extract the graph structure.
        
        Args:
            pbtxt_path: Path to the .pbtxt file
            
        Returns:
            (MediaPipeGraph object representing the parsed graph, header comment as inline_pbtxt_comment)
        """

        with open(pbtxt_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Extract header comment (first non-empty comment line at the top)
        header_comment = None
        for line in lines:
            if line.strip().startswith('#'):
                header_comment = line.strip()[1:].strip()
                break
            elif line.strip() != '':
                break  # Stop if first non-empty line is not a comment

        graph = MediaPipeGraph(
            input_streams=[],
            output_streams=[],
            input_side_packets=[],
            output_side_packets=[],
            nodes=[],
            packet_generators=[]
        )
        
        # Parse top-level streams and packets
        # Find the first node block to separate top-level fields
        node_start_idx = None
        for idx, line in enumerate(lines):
            if re.match(r'\s*node\s*\{', line):
                node_start_idx = idx
                break
        # Only consider lines before the first node block for graph-level fields
        top_level_content = '\n'.join(lines[:node_start_idx]) if node_start_idx is not None else content
        graph.input_streams = self._extract_pbtxt_field(top_level_content, "input_stream")
        graph.output_streams = self._extract_pbtxt_field(top_level_content, "output_stream")
        graph.input_side_packets = self._extract_pbtxt_field(top_level_content, "input_side_packet")
        graph.output_side_packets = self._extract_pbtxt_field(top_level_content, "output_side_packet")

        # Parse nodes
        graph.nodes = self._parse_nodes(content, lines)
        
        # Parse packet generators
        graph.packet_generators = self._parse_packet_generators(content)
        
        return graph, header_comment

    def extract_stream_fields(self, content: str, field_name: str) -> list:
        """Extract list field values for a given field from any content (pbtxt or comment block)."""
        pattern = rf'{field_name}\s*:\s*"([^"]*)"'
        return re.findall(pattern, content)

    def _extract_pbtxt_field(self, content: str, field_name: str) -> List[str]:
        """(Deprecated) Use extract_stream_fields instead."""
        return self.extract_stream_fields(content, field_name)

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
        """Parse a single node block, including extracting comment block above as description."""
        # Extract comment block above node
        description_lines = []
        i = start_line - 1
        while i >= 0 and lines[i].strip().startswith('#'):
            description_lines.insert(0, lines[i].strip()[1:].strip())
            i -= 1
        # Squashes a comment block into a single line
        def squash_comment_lines(lines):
            result = ''
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                if result:
                    result += ' '
                result += line
            return result.strip()
        inline_pbtxt_comment = squash_comment_lines(description_lines) if description_lines else None
        description = inline_pbtxt_comment
        # Find the end of this node block
        brace_count = 0
        end_line = start_line
        for j in range(start_line, len(lines)):
            line = lines[j]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and j > start_line:
                end_line = j
                break
        node_content = '\n'.join(lines[start_line:end_line+1])
        # Extract node fields from pbtxt
        name = self._extract_field(node_content, "calculator")
        if not name:
            return None
        input_streams = self.extract_stream_fields(node_content, "input_stream")
        output_streams = self.extract_stream_fields(node_content, "output_stream")
        input_side_packets = self.extract_stream_fields(node_content, "input_side_packet")
        output_side_packets = self.extract_stream_fields(node_content, "output_side_packet")
        # Extract node fields from comment block if not present in pbtxt
        comment_block = '\n'.join(description_lines)
        if not input_streams:
            input_streams = self.extract_stream_fields(comment_block, "input_stream")
        if not output_streams:
            output_streams = self.extract_stream_fields(comment_block, "output_stream")
        if not input_side_packets:
            input_side_packets = self.extract_stream_fields(comment_block, "input_side_packet")
        if not output_side_packets:
            output_side_packets = self.extract_stream_fields(comment_block, "output_side_packet")
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
            'line_number': start_line + 1,
            'description': description,
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


    def analyze_pipeline(self, pipeline_name: str, parent_output_dir: Optional[Path] = None, node_fields: dict = None) -> PipelineNode:

        """Analyze the specific hand landmark tracking pipeline and build a hierarchy tree."""

        if pipeline_name in self.graph_source_files:
            src_info = self.graph_source_files[pipeline_name]
            graph_path = str(src_info[0])
            graph_line_number = src_info[1]
            graph_line_code = src_info[2]
            self.print_with_ident(f'analyzing the pipeline definition of \'{pipeline_name}\' found at {Path(graph_path).relative_to(Path.cwd())}')
        else:
            raise FileNotFoundError(f'could not find a pipeline graph definition of {pipeline_name} in mediapipe source code')

        if graph_path.endswith('.pbtxt'):
            graph, graph_header_comment = self.parse_pbtxt_file(graph_path)
            graph_self_description = GraphSelfDescription(
                description=graph_header_comment,
                input_streams=graph.input_streams,
                output_streams=graph.output_streams,
                input_side_packets=graph.input_side_packets,
                output_side_packets=graph.output_side_packets,
                node_options=None
            )
        else:
            warning = '⚠️ this graph is defined in C++ code (not by a parseable .pbtxt file). its own nodes are therefore not expanded here, but you can read them in its source code.'
            self.print_with_ident(warning)
            return PipelineNode(
                name=pipeline_name,
                node_type='graph',
                source=graph_path,
                warning = warning,
                children=[],
                source_line_number=graph_line_number,
                source_line_code=graph_line_code,
                graph_self_description=None,
                description=None,
                input_streams=None,
                output_streams=None,
                input_side_packets=None,
                output_side_packets=None,
                node_options=None
            )
        calculator_mapping = dict[str, Path]()
        subgraph_mapping = dict[str, Path]()
        child_nodes = []
        for node in graph.nodes:
            node_name = getattr(node, 'name', None) or node.get('name')
            input_streams = getattr(node, 'input_streams', None) or node.get('input_streams')
            output_streams = getattr(node, 'output_streams', None) or node.get('output_streams')
            input_side_packets = getattr(node, 'input_side_packets', None) or node.get('input_side_packets')
            output_side_packets = getattr(node, 'output_side_packets', None) or node.get('output_side_packets')
            node_options = getattr(node, 'node_options', None) or node.get('node_options')
            node_description = node.get('description')
            warning = None
            source_line_number = None
            source_line_code = None
            node_source = None
            is_calc = is_graph = False
            node_source_rel = None

            # handle special cases filled by hand, as our current parsing implementation won't resolve them.
            # (though this can fully be automated to be extracted generically with a couple of hours' work)
            match node_name:
                case 'InferenceCalculator':
                    warning = (
                        f' *️this calculator has multiple implementions one per each target inference stack (e.g. cpu, xnnpack, and more). '
                        f'all implementations of it run a neural network ℹ️ see also https://ai.google.dev/edge/mediapipe/framework/framework_concepts/graphs_cpp '
                        'assume the one selected at runtime matches the target inference type used by the pipeline being run. '
                        'the current source association of it is hard-wired and not auto-discovered. ')
                    is_calc = True
                    node_source_rel = 'mediapipe/calculators/tensor/inference_calculator_cpu.cc'
                    node_source = str(Path.cwd() / node_source_rel)
                    source_line_number = None
                case 'LandmarkProjectionCalculator':
                    is_calc = True
                    warning = (
                        f' *️this calculator is defined in the source tree by C++ code (not in a pipeline\'s .pbtxt definition file) per pipeline each '
                        f'defined by the C++ api call graph.AddNode("\'{node_name}\'"). the current source association of it is hard-wired and not auto-discovered. ')
                    node_source_rel = 'mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc'
                    node_source = str(Path.cwd() / node_source_rel)
                    source_line_number = 344
                case 'WorldLandmarkProjectionCalculator':
                    is_calc = True
                    warning = (
                        f' *️this calculator is defined in the source tree by C++ code (not in a pipeline\'s .pbtxt definition file) per pipeline each '
                        f'defined by the C++ api call graph.AddNode("\'{node_name}\'") the current source association of it is hard-wired and not auto-discovered. ')
                    node_source_rel = 'mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc'
                    node_source = str(Path.cwd() / node_source_rel)
                    source_line_number = 356
            is_calc = is_calc or node_name in self.calculators_source_mapping
            is_graph = is_graph or node_name in self.graph_source_files
            if is_calc and is_graph:
                self.print_with_ident(f'❌️️️ node \'{node_name}\' is found to be both a calculator and a graph; this case is not currently handled downstream!')
            if not is_calc and not is_graph:
                self.print_with_ident(f'❌️️️ node \'{node_name}\' is neither a calculator nor a graph, which is not expected.')
                continue
            if is_calc:
                node_type = 'calculator'
                if node_name in self.calculators_source_mapping:
                    src_info = self.calculators_source_mapping[node_name]
                    node_source = str(src_info[0])
                    source_line_number = src_info[1]
                    source_line_code = src_info[2]
                    node_source_rel = str(Path(node_source).relative_to(Path.cwd())) if node_source else node_source
                if node_source_rel:
                    calculator_mapping[node_name] = node_source
                    self.print_with_ident(f'✅ calculator node \'{node_name}\' : {node_source_rel}')
                else:
                    self.print_with_ident(f'❌ could not locate the source code for node \'{node_name}\'')
                child_nodes.append(PipelineNode(
                    name=node_name,
                    node_type=node_type,
                    source=node_source,
                    warning=warning,
                    description=node_description,
                    children=[],
                    source_line_number=source_line_number,
                    source_line_code=source_line_code,
                    input_streams=input_streams,
                    output_streams=output_streams,
                    input_side_packets=input_side_packets,
                    output_side_packets=output_side_packets,
                    node_options=node_options,
                    graph_self_description=None
                ))
            elif is_graph:
                src_info = self.graph_source_files[node_name]
                node_source = str(src_info[0])
                source_line_number = src_info[1]
                source_line_code = src_info[2]
                subgraph_mapping[node_name] = node_source
                node_source_rel = str(Path(node_source).relative_to(Path.cwd())) if node_source else node_source
                self.print_with_ident(f'✅ graph node \'{node_name}\' : {node_source_rel}')
                self.print_with_ident(f'🔁 \'{node_name}\' is a graph')
                self.ident += 1
                # For subgraph nodes, pass node-level fields from parent node block
                child_node = self.analyze_pipeline(node_name, parent_output_dir, node_fields={
                    'input_streams': input_streams,
                    'output_streams': output_streams,
                    'input_side_packets': input_side_packets,
                    'output_side_packets': output_side_packets,
                    'node_options': node_options,
                    'description': node_description
                })
                child_nodes.append(child_node)
                self.ident -= 1
            if warning:
                self.print_with_ident(f'{warning}')
        # For the topmost graph node, do not assign a node description or node-level fields
        if node_fields is None:
            return PipelineNode(
                name=pipeline_name,
                node_type='graph',
                source=graph_path,
                warning=None,
                children=child_nodes,
                source_line_number=graph_line_number,
                source_line_code=graph_line_code,
                input_streams=None,
                output_streams=None,
                input_side_packets=None,
                output_side_packets=None,
                node_options=None,
                description=None,
                graph_self_description=graph_self_description
            )
        else:
            # For subgraph nodes, assign both node-level fields and graph_self_description
            return PipelineNode(
                name=pipeline_name,
                node_type='graph',
                source=graph_path,
                warning=None,
                children=child_nodes,
                source_line_number=graph_line_number,
                source_line_code=graph_line_code,
                input_streams=node_fields.get('input_streams'),
                output_streams=node_fields.get('output_streams'),
                input_side_packets=node_fields.get('input_side_packets'),
                output_side_packets=node_fields.get('output_side_packets'),
                node_options=node_fields.get('node_options'),
                description=node_fields.get('description'),
                graph_self_description=graph_self_description
            )

    def write_pipeline_outputs(self, root_node: PipelineNode, output_dir: Path):
        """Write the pipeline hierarchy to markdown (with and without line numbers), JSON, YAML, and HTML files in output_dir."""
        # Group outputs by format
        md_dir = output_dir / 'markdown'
        json_dir = output_dir / 'json'
        yaml_dir = output_dir / 'yaml'
        html_dir = output_dir / 'html'
        md_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        yaml_dir.mkdir(parents=True, exist_ok=True)
        html_dir.mkdir(parents=True, exist_ok=True)
        # Write JSON (basic)
        json_basic_path = json_dir / 'pipeline.basic.json'
        with open(json_basic_path, 'w') as f:
            json.dump(self._node_to_dict_rel(root_node), f, indent=2)
        # Write YAML (basic)
        yaml_basic_path = yaml_dir / 'pipeline.basic.yaml'
        with open(yaml_basic_path, 'w') as f:
            yaml.dump(self._node_to_dict_rel(root_node), f, sort_keys=False, allow_unicode=True)
        # Write Markdown (basic, with line numbers)
        md_basic_path = md_dir / 'pipeline.basic.md'
        with open(md_basic_path, 'w') as f:
            f.write(self._node_to_markdown_with_links(root_node, 0, json_basic_path, yaml_basic_path, True, script_name='pipeline_parser.py', use_absolute_links=True))
        # Write Markdown (basic, no line numbers)
        md_basic_nolines_path = md_dir / 'pipeline.basic.nolines.md'
        with open(md_basic_nolines_path, 'w') as f:
            f.write(self._node_to_markdown_with_links(root_node, 0, json_basic_path, yaml_basic_path, False, script_name='pipeline_parser.py', use_absolute_links=True))
        # Write JSON (verbose)
        json_verbose_path = json_dir / 'pipeline.verbose.json'
        with open(json_verbose_path, 'w') as f:
            json.dump(self._node_to_dict_verbose(root_node), f, indent=2)
        # Write YAML (verbose)
        yaml_verbose_path = yaml_dir / 'pipeline.verbose.yaml'
        with open(yaml_verbose_path, 'w') as f:
            yaml.dump(self._node_to_dict_verbose(root_node), f, sort_keys=False, allow_unicode=True)
        # Write Markdown (verbose, with line numbers)
        md_verbose_path = md_dir / 'pipeline.verbose.md'
        with open(md_verbose_path, 'w') as f:
            f.write(self._node_to_markdown_with_links_verbose(root_node, 0, json_verbose_path, yaml_verbose_path, md_basic_path, md_basic_nolines_path, json_basic_path, yaml_basic_path, True, md_verbose_path, yaml_verbose_path, json_verbose_path, script_name='pipeline_parser.py', use_absolute_links=True))
        # Write Markdown (verbose, no line numbers)
        md_verbose_nolines_path = md_dir / 'pipeline.verbose.nolines.md'
        with open(md_verbose_nolines_path, 'w') as f:
            f.write(self._node_to_markdown_with_links_verbose(root_node, 0, json_verbose_path, yaml_verbose_path, md_basic_path, md_basic_nolines_path, json_basic_path, yaml_basic_path, False, md_verbose_nolines_path, yaml_verbose_path, json_verbose_path, script_name='pipeline_parser.py', use_absolute_links=True))
        # Write HTML (basic)
        html_basic_path = html_dir / 'pipeline.basic.html'
        with open(html_basic_path, 'w') as f:
            f.write(self._node_to_html_with_hover(root_node, 0, script_name='pipeline_parser.py'))
        # Store output paths for printing
        self._output_paths = {
            'md_basic': md_basic_path,
            'md_basic_nolines': md_basic_nolines_path,
            'md_verbose': md_verbose_path,
            'json_basic': json_basic_path,
            'json_verbose': json_verbose_path,
            'yaml_basic': yaml_basic_path,
            'yaml_verbose': yaml_verbose_path,
            'html_basic': html_basic_path,
        }

    def _node_to_html_with_hover(self, node: PipelineNode, level: int, script_name='pipeline_parser.py') -> str:
        """Render the pipeline as an HTML tree with hover tooltips for node descriptions."""
        # Helper for HTML escaping
        import html
        def esc(s):
            return html.escape(s) if s else ''
        # Helper for hyperlinks (no line numbers)
        def make_link(node):
            if node.node_type == 'calculator' and node.source:
                path = str(node.source)
                html_file_dir = Path('mediapipe_analysis/analysis/output/html').resolve()
                try:
                    rel_path = os.path.relpath(Path(path).resolve(), html_file_dir)
                except Exception:
                    rel_path = str(Path(path).name)
                return esc(rel_path)
            elif node.source:
                path = str(node.source)
                html_file_dir = Path('mediapipe_analysis/analysis/output/html').resolve()
                try:
                    rel_path = os.path.relpath(Path(path).resolve(), html_file_dir)
                except Exception:
                    rel_path = str(Path(path).name)
                return esc(rel_path)
            return None
        # Recursive rendering
        def render_node(node, level):
            indent = '    ' * level
            link = make_link(node)
            desc = esc(node.description) if node.description else ''
            warning = esc(node.warning) if node.warning else ''
            hover = ''
            if desc or warning or node.input_streams or node.output_streams or node.input_side_packets or node.output_side_packets or (node.node_options and node.node_options != {}):
                hover = '<div class="hoverbox">'
                if desc:
                    hover += f'<div class="hoverbox-desc">{desc}</div>'
                # Add table for fields (above warning)
                def table_row(label, items):
                    if items:
                        if isinstance(items, list):
                            values = '<br>'.join([esc(str(s)) for s in items])
                        else:
                            values = esc(str(items))
                        return f'<tr><td class="hoverbox-label">{label}</td><td class="hoverbox-value">{values}</td></tr>'
                    return ''
                table_rows = ''
                table_rows += table_row('input streams', getattr(node, 'input_streams', None))
                table_rows += table_row('output streams', getattr(node, 'output_streams', None))
                table_rows += table_row('input side packets', getattr(node, 'input_side_packets', None))
                table_rows += table_row('output side packets', getattr(node, 'output_side_packets', None))
                if node.node_options and node.node_options != {}:
                    opts = [f'{esc(str(k))}: {esc(str(v))}' for k, v in node.node_options.items()]
                    table_rows += table_row('node options', opts)
                table_html = ''
                if table_rows:
                    table_html = '<br><table class="hoverbox-fields">' + table_rows + '</table>'
                if table_html:
                    hover += table_html
                if table_html and warning:
                    hover += '<hr style="margin:4px 0;">'
                if warning:
                    hover += f'<span style="color:gray;font-weight:bold;">{warning}</span>'
                hover += '</div>'
            has_warning_class = ' has-warning' if warning else ''
            node_html = ''

            display_name = esc(node.name)
            if node.node_type == 'graph':
                display_name = f'🔁 {display_name} (graph)'
            if link:
                node_html += f'{indent}<span class="node-container{has_warning_class}"><a href="{link}">{display_name}</a>{hover}</span>'
            else:
                node_html += f'{indent}<span class="node-container{has_warning_class}">{display_name}{hover}</span>'
            if node.children:
                node_html += '\n' + indent + '<ul>\n'
                for child in node.children:
                    node_html += indent + '  <li>' + render_node(child, level + 1) + '</li>\n'
                node_html += indent + '</ul>'
            return node_html
        # HTML header, styles, and table
        html_head = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline {name}</title>
<style>
body {{ font-family: sans-serif; background: #111; color: #eee; }}
ul {{ list-style-type: none; }}
/* Hierarchy links: green shade, bold, no underline */
ul > li > .node-container > a, ul > li > .node-container {{ color: #7CFC00; font-weight: bold; }}
ul > li > .node-container > a {{ text-decoration: none; }}
ul, li {{ line-height: 1.7; }}
table a {{ color: #4fc3f7; text-decoration: underline; }}
.node-container {{ position: relative; }}
.hoverbox {{
  display: none;
  position: absolute;
  left: 100%;
  top: 0;
  z-index: 10;
  background: #222;
  color: #eee;
  border: 1px solid #555;
  padding: 8px;
  min-width: 400px;
  width: auto;
  box-shadow: 2px 2px 8px #222;
  white-space: pre-line;
}}
.node-container:hover .hoverbox {{
  display: block;
}}
.hoverbox-fields {{
  width: auto;
  border-collapse: collapse;
  margin-top: 4px;
  margin-bottom: 4px;
  table-layout: auto;
}}
.hoverbox-fields td {{
  padding: 2px 8px 2px 0;
  vertical-align: top;
}}
.hoverbox-label {{
  color: #4fc3f7;
  font-weight: bold;
  text-align: right;
  min-width: 160px;
  white-space: nowrap;
}}
.hoverbox-value {{
  color: #ffd700;
  text-align: left;
  white-space: nowrap;
  overflow-x: auto;
  text-overflow: ellipsis;
}}
hr {{ border: none; border-top: 1px solid #555; margin: 6px 0; }}
.has-warning a {{ color: #b0b0b0 !important; }}
.has-warning {{ color: #b0b0b0 !important; }}
code, pre {{ background: #222; color: #eee; }}
.pipeline-info, .formats-table, .footer-note {{ color: #aaa; }}
</style>
</head>
<body>
'''.format(name=esc(node.name))
        html_tail = f'<br><br>{self._formats_table_html(script_name)}<br><div class="footer-note" style="color:#aaa;font-size:small;">To regenerate this analysis for the mediapipe directory, run <code>{script_name}</code>.<br>For best navigation, open this HTML file via a local web server (e.g. <code>python3 -m http.server</code>) from your workspace root.</div>\n</body>\n</html>'
        html_body = f'<h1>Pipeline {esc(node.name)}</h1>\n<ul>\n<li>{render_node(node, 0)}</li>\n</ul>'
        return html_head + html_body + html_tail

    def _formats_table_html(self, script_name):
        # HTML version of the formats table, with reference to the HTML format
        return (
            '<h2>More Formats</h2>'
            '<table border="1" cellpadding="4" cellspacing="0">'
            '<tr><th>Format</th><th>Description</th></tr>'
            '<tr><td><a href="../markdown/pipeline.basic.md">Basic Markdown (with line numbers)</a></td><td>Tree with hyperlinks to each node\'s source (with line numbers)</td></tr>'
            '<tr><td><a href="../markdown/pipeline.basic.nolines.md">Basic Markdown (no line numbers)</a></td><td>Tree with hyperlinks to each node\'s source (no line numbers)</td></tr>'
            '<tr><td><a href="../markdown/pipeline.verbose.md">Verbose Markdown</a></td><td>Tree with more node fields (streams, packets, options)</td></tr>'
            '<tr><td><a href="../html/pipeline.basic.html">HTML</a></td><td>HTML format with node descriptions upon hover</td></tr>'
            '<tr><td><a href="../json/pipeline.basic.json">Basic JSON</a></td><td>Tree in JSON format</td></tr>'
            '<tr><td><a href="../yaml/pipeline.basic.yaml">Basic YAML</a></td><td>Tree in YAML format</td></tr>'
            '<tr><td><a href="../json/pipeline.verbose.json">Verbose JSON</a></td><td>Tree with more node fields (streams, packets, options)</td></tr>'
            '<tr><td><a href="../yaml/pipeline.verbose.yaml">Verbose YAML</a></td><td>Tree with more node fields (streams, packets, options)</td></tr>'
            '</table>'
        )

    def _formats_table(self, abs_dir, script_name):
        # Helper to generate the formats table for markdown (relative links to grouped outputs)
        return (
            '# More Formats\n\n'
            '| Format | Description |\n'
            '|--------|-------------|\n'
            f'| [Basic Markdown (with line numbers)](pipeline.basic.md) | Tree with hyperlinks to each node\'s source (with line numbers) |\n'
            f'| [Basic Markdown (no line numbers)](pipeline.basic.nolines.md) | Tree with hyperlinks to each node\'s source (no line numbers) |\n'
            f'| [Verbose Markdown](pipeline.verbose.md) | Tree with more node fields (streams, packets, options) |\n'
            f'| [HTML](../html/pipeline.basic.html) | HTML format with node descriptions upon hover |\n'
            f'| [Basic JSON](../json/pipeline.basic.json) | Tree in JSON format |\n'
            f'| [Basic YAML](../yaml/pipeline.basic.yaml) | Tree in YAML format |\n'
            f'| [Verbose JSON](../json/pipeline.verbose.json) | Tree with more node fields (streams, packets, options) |\n'
            f'| [Verbose YAML](../yaml/pipeline.verbose.yaml) | Tree with more node fields (streams, packets, options) |\n'
            f'To regenerate this analysis for the mediapipe directory, run `{script_name}`.\n'
        )

    def _node_to_dict_rel(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary, making 'source' fields relative to cwd, and using 'nodes' instead of 'children'."""
        def rel(p):
            if p is None:
                return None
            try:
                return str(Path(p).relative_to(Path.cwd()))
            except Exception:
                return str(p)
        d = {
            'name': node.name,
            'type': node.node_type,
            'source': rel(node.source),
            'description': node.description,
        }
        if node.warning:
            d['warning'] = node.warning
        if node.children:
            d['nodes'] = [self._node_to_dict_rel(child) for child in node.children]
        return d

    def _node_to_dict_verbose(self, node: PipelineNode) -> dict:
        """Recursively convert a PipelineNode tree to a dictionary for verbose serialization (all fields), omitting empty stream/packet/options fields."""
        d = {
            'name': node.name,
            'type': node.node_type,
            'source': node.source,
            'description': node.description,
            'source_line_number': node.source_line_number,
        }
        if node.warning:
            d['warning'] = node.warning
        if node.input_streams:
            d['input_streams'] = node.input_streams
        if node.output_streams:
            d['output_streams'] = node.output_streams
        if node.input_side_packets:
            d['input_side_packets'] = node.input_side_packets
        if node.output_side_packets:
            d['output_side_packets'] = node.output_side_packets
        if node.node_options and node.node_options != {}:
            d['node_options'] = node.node_options
        # Add graph_self_description if present, omitting empty/null fields
        if node.graph_self_description:
            gsd = {}
            if node.graph_self_description.description:
                gsd['description'] = node.graph_self_description.description
            if node.graph_self_description.input_streams:
                if isinstance(node.graph_self_description.input_streams, list) and node.graph_self_description.input_streams:
                    gsd['input_streams'] = node.graph_self_description.input_streams
            if node.graph_self_description.output_streams:
                if isinstance(node.graph_self_description.output_streams, list) and node.graph_self_description.output_streams:
                    gsd['output_streams'] = node.graph_self_description.output_streams
            if node.graph_self_description.input_side_packets:
                if isinstance(node.graph_self_description.input_side_packets, list) and node.graph_self_description.input_side_packets:
                    gsd['input_side_packets'] = node.graph_self_description.input_side_packets
            if node.graph_self_description.output_side_packets:
                if isinstance(node.graph_self_description.output_side_packets, list) and node.graph_self_description.output_side_packets:
                    gsd['output_side_packets'] = node.graph_self_description.output_side_packets
            if node.graph_self_description.node_options and node.graph_self_description.node_options != {}:
                gsd['node_options'] = node.graph_self_description.node_options
            if gsd:
                d['graph_self_description'] = gsd
        if node.children:
            d['nodes'] = [self._node_to_dict_verbose(child) for child in node.children]
        return d

    def _node_to_markdown_with_links(self, node: PipelineNode, level: int, json_path: Path, yaml_path: Path, with_line_numbers: bool, script_name='pipeline_parser.py', use_absolute_links=False) -> str:
        indent = '    ' * level
        if level == 0:
            abs_dir = Path(json_path).parent.resolve()
            md = f'# Pipeline {node.name}\n\n'
        else:
            md = ''
        def relpath(p):
            if p is None:
                return ''
            try:
                if use_absolute_links:
                    return str(Path(p).resolve())
                md_dir = Path(json_path).parent.resolve()
                return os.path.relpath(str(Path(p).resolve()), md_dir)
            except Exception:
                return str(Path(p).name)
        # Prepare display name
        display_name = node.name
        if node.node_type == 'graph':
            display_name += ' (graph)'
        # Always render node as hyperlink
        if node.node_type == 'calculator' and node.source and node.source_line_number:
            if with_line_numbers:
                link = f"{relpath(node.source)}#L{node.source_line_number}"
                md += f'{indent}- [{display_name}]({link})'
            else:
                link = relpath(node.source)
                md += f'{indent}- [{display_name}]({link})'
        elif node.source:
            link = relpath(node.source)
            md += f'{indent}- [{display_name}]({link})'
        else:
            # If no source, just link to a placeholder (could be improved)
            md += f'{indent}- [{display_name}](#)'
        # Append warning if present
        if node.warning:
            md += f': {node.warning}'
        md += '\n'
        for child in node.children:
            md += self._node_to_markdown_with_links(child, level + 1, json_path, yaml_path, with_line_numbers, script_name=script_name, use_absolute_links=use_absolute_links)
        if level == 0:
            md += '\n' + self._formats_table(abs_dir, script_name)
        return md

    def _node_to_markdown_with_links_verbose(self, node: PipelineNode, level: int, json_verbose_path: Path, yaml_verbose_path: Path, md_basic_path: Path, md_basic_nolines_path: Path, json_basic_path: Path, yaml_basic_path: Path, with_line_numbers: bool, md_verbose_path: Path, yaml_verbose_path2: Path, json_verbose_path2: Path, script_name='pipeline_parser.py', use_absolute_links=False) -> str:
        indent = '    ' * level
        table_indent = '    ' * (level + 1)
        if level == 0:
            abs_dir = Path(json_verbose_path).parent.resolve()
            md = f'# Pipeline Hierarchy for `{node.name}` (verbose)\n\n'
        else:
            md = ''
        def relpath(p):
            if p is None:
                return ''
            try:
                if use_absolute_links:
                    return str(Path(p).resolve())
                md_dir = Path(md_verbose_path).parent.resolve()
                return os.path.relpath(str(Path(p).resolve()), md_dir)
            except Exception:
                return str(Path(p).name)
        display_name = node.name
        if node.node_type == 'graph':
            display_name += ' (graph)'
        # Render node as hyperlink
        if node.node_type == 'calculator' and node.source and node.source_line_number:
            if with_line_numbers:
                link = f"{relpath(node.source)}#L{node.source_line_number}"
                md += f'{indent}- [{display_name}]({link})'
            else:
                link = relpath(node.source)
                md += f'{indent}- [{display_name}]({link})'
        elif node.source:
            link = relpath(node.source)
            md += f'{indent}- [{display_name}]({link})'
        else:
            md += f'{indent}- [{display_name}](#)'
        # Append warning if present
        if node.warning:
            md += f'{indent}  {node.warning}\n'
        # HTML table for fields
        table_rows = []
        def html_row(header, items):
            if not items:
                return ''
            values = '<br>'.join(str(s) for s in items)
            return f'{table_indent}<tr><td><i>{header}</i></td><td>{values}</td></tr>\n'
        if node.input_streams:
            table_rows.append(html_row('input streams', node.input_streams))
        if node.output_streams:
            table_rows.append(html_row('output streams', node.output_streams))
        if node.input_side_packets:
            table_rows.append(html_row('input side packets', node.input_side_packets))
        if node.output_side_packets:
            table_rows.append(html_row('output side packets', node.output_side_packets))
        if node.node_options and node.node_options != {}:
            import json as _json
            node_opts_str = _json.dumps(node.node_options, indent=2)
            node_opts_lines = node_opts_str.split('\n')
            values = '<br>'.join(node_opts_lines)
            table_rows.append(f'{table_indent}<tr><td><i>node options</i></td><td>{values}</td></tr>\n')
        # Add graph_self_description details for graph nodes
        if node.node_type == 'graph' and node.graph_self_description:
            md += f'\n{indent}  * Graph Description: {node.graph_self_description.description}'
            if node.graph_self_description.input_streams:
                md += f'\n{indent}  * Input Streams: {node.graph_self_description.input_streams}'
            if node.graph_self_description.output_streams:
                md += f'\n{indent}  * Output Streams: {node.graph_self_description.output_streams}'
            if node.graph_self_description.input_side_packets:
                md += f'\n{indent}  * Input Side Packets: {node.graph_self_description.input_side_packets}'
            if node.graph_self_description.output_side_packets:
                md += f'\n{indent}  * Output Side Packets: {node.graph_self_description.output_side_packets}'
            if node.graph_self_description.node_options:
                md += f'\n{indent}  * Node Options: {node.graph_self_description.node_options}'
        md += '\n'
        for child in node.children:
            md += self._node_to_markdown_with_links_verbose(child, level + 1, json_verbose_path, yaml_verbose_path, md_basic_path, md_basic_nolines_path, json_basic_path, yaml_basic_path, with_line_numbers, md_verbose_path, yaml_verbose_path2, json_verbose_path2, script_name=script_name, use_absolute_links=use_absolute_links)
        if level == 0:
            md += '\n' + self._formats_table(abs_dir, script_name)
        return md

    @staticmethod
    def parse_stream_desc(desc: str) -> dict:
        """ Parses a stream description into its components: name, tag (if present), and clone (if present).
        
        CLONE:number:name is parsed as clone=number, name=name, no tag.
        CLONE:name is parsed as clone=name, name=name, no tag.
        TAG:name is parsed as tag=TAG, name=name.
        name alone is parsed as name=name, no tag or clone. """

        clone = None
        tag = None
        name = None
        parts = desc.split(':')
        if len(parts) == 3 and parts[0] == 'CLONE':
            clone = parts[1]
            name = parts[2]
        elif len(parts) == 2 and parts[0] == 'CLONE':
            clone = parts[1]
            name = parts[1]
        elif len(parts) == 2:
            tag = parts[0]
            name = parts[1]
        elif len(parts) == 1:
            name = parts[0]
        else:
            # fallback: treat last part as name, second-to-last as tag if present
            name = parts[-1]
            if len(parts) > 1:
                tag = parts[-2]
        return {'name': name, 'tag': tag, 'clone': clone}

    @staticmethod
    def streams_match(desc1: str, desc2: str) -> bool:
        """Return True if two stream descriptions match by name or by tag (if both have tags)."""
        s1 = MediaPipePipelineParser.parse_stream_desc(desc1)
        s2 = MediaPipePipelineParser.parse_stream_desc(desc2)
        if s1['name'] == s2['name']:
            return True
        if s1['tag'] and s2['tag'] and s1['tag'] == s2['tag']:
            return True
        return False

    def check_graph_node_streams(self, graph_node: PipelineNode, parent_node_fields: dict) -> dict:
        """Check that every stream defined for the graph as a node matches exactly one field self-defined by the graph itself, and vice versa, for all stream types."""
        results = {}
        for field in ['input_streams', 'output_streams', 'input_side_packets', 'output_side_packets']:
            parent_streams = parent_node_fields.get(field) or []
            graph_streams = getattr(graph_node, field) or []
            # Check each parent stream matches exactly one graph stream
            unmatched_parent = []
            for ps in parent_streams:
                matches = [gs for gs in graph_streams if self.streams_match(ps, gs)]
                if len(matches) != 1:
                    unmatched_parent.append(ps)
            # Check each graph stream matches exactly one parent stream
            unmatched_graph = []
            for gs in graph_streams:
                matches = [ps for ps in parent_streams if self.streams_match(gs, ps)]
                if len(matches) != 1:
                    unmatched_graph.append(gs)
            results[field] = {
                'unmatched_parent': unmatched_parent,
                'unmatched_graph': unmatched_graph,
                'all_matched': not unmatched_parent and not unmatched_graph
            }
        return results

def main():
    parser = MediaPipePipelineParser(Path('mediapipe').resolve())
    print()
    root_node = parser.analyze_pipeline('HandLandmarkTrackingCpu')
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path('mediapipe_analysis/analysis/output')
    parser.write_pipeline_outputs(root_node, output_dir)
    output_dir_abs = output_dir.resolve()
    md_dir = output_dir_abs / 'markdown'
    json_dir = output_dir_abs / 'json'
    yaml_dir = output_dir_abs / 'yaml'
    html_dir = output_dir_abs / 'html'
    outputs = [
        (html_dir / "pipeline.basic.html", "HTML format with node descriptions from pipeline inline documentation (hover for details); also includes free text descriptions of the nodes"),
        (md_dir / "pipeline.basic.nolines.md", "Tree with hyperlinks to each node's source"),
        (md_dir / "pipeline.basic.md", "Tree with hyperlinks to each node's source (with also line numbers included in source file hyperlinks)"),
        (md_dir / "pipeline.verbose.md", "Tree with more node fields (streams, packets, options), hyperlinks to each node's source (with line numbers)"),
        (md_dir / "pipeline.verbose.nolines.md", "Tree with more node fields (streams, packets, options), hyperlinks to each node's source (no line numbers)"),
        (json_dir / "pipeline.basic.json", "Tree in JSON format"),
        (json_dir / "pipeline.verbose.json", "Tree with more node fields (streams, packets, options) in JSON format"),
        (yaml_dir / "pipeline.basic.yaml", "Tree in YAML format"),
        (yaml_dir / "pipeline.verbose.yaml", "Tree with more node fields (streams, packets, options) in YAML format"),
    ]
    # Find max path length for alignment
    maxlen = max(len(f"file://{str(path)}") for path, _ in outputs)
    # Print as table by format
    print(f"\nthe pipeline analysis output shown above is also available in multiple formats and levels of verbosity under file://{output_dir_abs}/.\n"
          f"use the html/markdown variants for quick navigation to source files, and the data formats for machine consumption:\n")
    print("html:")
    print(f"  {'file://' + str(outputs[0][0]):<{maxlen}}  :  {outputs[0][1]}")
    print("\nmarkdown:")
    for path, desc in outputs[1:5]:
        print(f"  {'file://' + str(path):<{maxlen}}  :  {desc}")
    print("\njson:")
    for path, desc in outputs[5:7]:
        print(f"  {'file://' + str(path):<{maxlen}}  :  {desc}")
    print("\nyaml:")
    for path, desc in outputs[7:]:
        print(f"  {'file://' + str(path):<{maxlen}}  :  {desc}")
    print()

if __name__ == "__main__":
    main()
