#!/usr/bin/env python3
"""
MediaPipe Pipeline Parser

Parses MediaPipe .pbtxt graph definition files to extract the computational
graph structure and identify all calculator nodes and their connections.
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import os

@dataclass
class CalculatorNode:
    """Represents a calculator node in the MediaPipe graph."""
    name: str
    calculator: str
    input_streams: List[str]
    output_streams: List[str]
    input_side_packets: List[str]
    output_side_packets: List[str]
    node_options: Dict[str, str]
    line_number: int

@dataclass
class SubGraph:
    """Represents a sub-graph within the MediaPipe graph."""
    type: str
    input_streams: List[str]
    output_streams: List[str]
    input_side_packets: List[str]
    output_side_packets: List[str]
    line_number: int

@dataclass
class MediaPipeGraph:
    """Represents the complete MediaPipe computation graph."""
    input_streams: List[str]
    output_streams: List[str]
    input_side_packets: List[str]
    output_side_packets: List[str]
    nodes: List[CalculatorNode]
    subgraphs: List[SubGraph]
    packet_generators: List[Dict[str, str]]

class MediaPipePipelineParser:
    def __init__(self, mediapipe_source_path: Path):
        """
        Initialize the parser with MediaPipe source path.
        
        Args:
            mediapipe_source_path: Path to MediaPipe source directory
        """
        self.mediapipe_source = Path(mediapipe_source_path)
        self.calculator_registry: Dict[str, Path] = {}
        self.graph_registry: Dict[str, Path] = {}
        
    def parse_pbtxt_file(self, pbtxt_path: Path) -> MediaPipeGraph:
        """
        Parse a .pbtxt file and extract the graph structure.
        
        Args:
            pbtxt_path: Path to the .pbtxt file
            
        Returns:
            MediaPipeGraph object representing the parsed graph
        """
        print(f"Parsing {pbtxt_path}")
        
        with open(pbtxt_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        graph = MediaPipeGraph(
            input_streams=[],
            output_streams=[],
            input_side_packets=[],
            output_side_packets=[],
            nodes=[],
            subgraphs=[],
            packet_generators=[]
        )
        
        # Parse top-level streams and packets
        graph.input_streams = self._extract_list_field(content, "input_stream")
        graph.output_streams = self._extract_list_field(content, "output_stream")
        graph.input_side_packets = self._extract_list_field(content, "input_side_packet")
        graph.output_side_packets = self._extract_list_field(content, "output_side_packet")
        
        # Parse nodes
        graph.nodes = self._parse_nodes(content, lines)
        
        # Parse subgraphs  
        graph.subgraphs = self._parse_subgraphs(content, lines)
        
        # Parse packet generators
        graph.packet_generators = self._parse_packet_generators(content)
        
        return graph
    
    def _extract_list_field(self, content: str, field_name: str) -> List[str]:
        """Extract list field values from pbtxt content."""
        pattern = rf'{field_name}\s*:\s*"([^"]*)"'
        matches = re.findall(pattern, content)
        return matches
    
    def _parse_nodes(self, content: str, lines: List[str]) -> List[CalculatorNode]:
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
    
    def _parse_single_node(self, lines: List[str], start_line: int) -> Optional[CalculatorNode]:
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
        name = self._extract_field(node_content, "name")
        calculator = self._extract_field(node_content, "calculator")
        
        if not calculator:
            return None
        
        input_streams = self._extract_list_field(node_content, "input_stream")
        output_streams = self._extract_list_field(node_content, "output_stream")
        input_side_packets = self._extract_list_field(node_content, "input_side_packet")
        output_side_packets = self._extract_list_field(node_content, "output_side_packet")
        
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
        
        return CalculatorNode(
            name=name or "",
            calculator=calculator,
            input_streams=input_streams,
            output_streams=output_streams,
            input_side_packets=input_side_packets,
            output_side_packets=output_side_packets,
            node_options=node_options,
            line_number=start_line + 1
        )
    
    def _parse_subgraphs(self, content: str, lines: List[str]) -> List[SubGraph]:
        """Parse subgraph definitions."""
        subgraphs = []
        
        # Find subgraph blocks
        subgraph_pattern = r'node\s*\{\s*calculator\s*:\s*"([^"]*Graph)"'
        
        for i, line in enumerate(lines):
            match = re.search(subgraph_pattern, line)
            if match:
                graph_type = match.group(1) + "Graph"
                
                # Find the full node block for this subgraph
                brace_count = 0
                start_line = i
                end_line = i
                
                # Find the start of the node block
                while start_line > 0 and 'node {' not in lines[start_line]:
                    start_line -= 1
                
                # Find the end of the node block
                for j in range(start_line, len(lines)):
                    line_content = lines[j]
                    brace_count += line_content.count('{') - line_content.count('}')
                    if brace_count == 0 and j > start_line:
                        end_line = j
                        break
                
                subgraph_content = '\n'.join(lines[start_line:end_line+1])
                
                input_streams = self._extract_list_field(subgraph_content, "input_stream")
                output_streams = self._extract_list_field(subgraph_content, "output_stream")
                input_side_packets = self._extract_list_field(subgraph_content, "input_side_packet")
                output_side_packets = self._extract_list_field(subgraph_content, "output_side_packet")
                
                subgraph = SubGraph(
                    type=graph_type,
                    input_streams=input_streams,
                    output_streams=output_streams,
                    input_side_packets=input_side_packets,
                    output_side_packets=output_side_packets,
                    line_number=start_line + 1
                )
                subgraphs.append(subgraph)
        
        return subgraphs
    
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
    
    def find_calculator_sources(self) -> Dict[str, List[Path]]:
        """Find all calculator C++ source files in MediaPipe."""
        calculator_files = {}
        
        # Search for calculator files
        for root, dirs, files in os.walk(self.mediapipe_source):
            for file in files:
                if file.endswith('_calculator.cc') or file.endswith('_calculator.h'):
                    path = Path(root) / file
                    calculator_name = file.replace('_calculator.cc', '').replace('_calculator.h', '')
                    calculator_name = self._camel_case(calculator_name)
                    
                    if calculator_name not in calculator_files:
                        calculator_files[calculator_name] = []
                    calculator_files[calculator_name].append(path)
        
        return calculator_files
    
    def find_graph_definitions(self) -> Dict[str, Path]:
        """Find all graph definition .pbtxt files."""
        graph_files = {}
        
        for root, dirs, files in os.walk(self.mediapipe_source):
            for file in files:
                if file.endswith('.pbtxt'):
                    path = Path(root) / file
                    graph_name = file.replace('.pbtxt', '')
                    graph_files[graph_name] = path
        
        return graph_files
    
    def _camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.capitalize() for x in components)
    
    def analyze_hand_pipeline(self) -> Dict:
        """Analyze the specific hand landmark tracking pipeline."""
        # Find the hand landmark tracking graph
        graph_files = self.find_graph_definitions()
        hand_graph_path = None
        
        for name, path in graph_files.items():
            if 'hand_landmark_tracking_cpu' in name:
                hand_graph_path = path
                break
        
        if not hand_graph_path:
            raise FileNotFoundError("Could not find hand_landmark_tracking_cpu.pbtxt")
        
        # Parse the main graph
        main_graph = self.parse_pbtxt_file(hand_graph_path)
        
        # Find calculator sources
        calculator_sources = self.find_calculator_sources()
        
        # Map calculators to their source files
        calculator_mapping = {}
        for node in main_graph.nodes:
            calc_name = node.calculator
            if calc_name in calculator_sources:
                calculator_mapping[calc_name] = calculator_sources[calc_name]
            else:
                # Try to find by partial match
                for source_name, paths in calculator_sources.items():
                    if calc_name.lower() in source_name.lower() or source_name.lower() in calc_name.lower():
                        calculator_mapping[calc_name] = paths
                        break
        
        return {
            'main_graph': main_graph,
            'hand_graph_path': hand_graph_path,
            'calculator_mapping': calculator_mapping,
            'all_calculator_sources': calculator_sources,
            'all_graph_files': graph_files
        }

def main():
    """Test the pipeline parser."""

    source_path = 'mediapipe'
    
    # Parse the hand pipeline
    parser = MediaPipePipelineParser(source_path)
    analysis = parser.analyze_hand_pipeline()

    print(f"Found hand pipeline at: {analysis['hand_graph_path']}")
    print(f"Main graph has {len(analysis['main_graph'].nodes)} nodes")
    print(f"Main graph has {len(analysis['main_graph'].subgraphs)} subgraphs")

    print("\nCalculator nodes:")
    for node in analysis['main_graph'].nodes:
        print(f"  - {node.calculator} (line {node.line_number})")

    print("\nSubgraphs:")
    for subgraph in analysis['main_graph'].subgraphs:
        print(f"  - {subgraph.type} (line {subgraph.line_number})")

    print(f"\nFound {len(analysis['calculator_mapping'])} calculator mappings")
    print(f"Found {len(analysis['all_calculator_sources'])} total calculator sources")


if __name__ == "__main__":
    main()