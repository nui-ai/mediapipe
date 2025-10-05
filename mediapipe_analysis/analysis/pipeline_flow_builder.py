"""
MediaPipe Pipeline Flow Builder

Builds a stream-level graph representation of MediaPipe pipelines where each node is
a (node_name, stream_name) pair, capturing the detailed flow of data through the pipeline.
This handles stream name translations at graph boundaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import copy
import re


class PipelineFlowBuilder:
    """Builds a stream-level graph representation of the pipeline."""

    def __init__(self, pipeline_parser):
        """Initialize with the parser to access stream matching functionality."""
        self.parser = pipeline_parser
        self.nodes = []  # List of (node_name, stream_name) tuples
        self.feeds = {}  # Mapping from (node, stream) to list of (node, stream) it feeds
        self.feeds_from = {}  # Mapping from (node, stream) to list of (node, stream) it feeds from

    def build_flow_graph(self, pipeline_json_path: Path) -> dict:
        """
        Build a flow graph from a parsed pipeline.verbose.json file.

        Args:
            pipeline_json_path: Path to the pipeline.verbose.json file

        Returns:
            Dictionary representing the flow graph
        """
        # Load pipeline JSON
        with open(pipeline_json_path, 'r') as f:
            pipeline = json.load(f)

        # Flatten the node hierarchy to get all nodes with their original names
        flat_nodes = self._flatten_nodes(pipeline)

        # Map output streams to producing nodes
        output_stream_to_node = {}
        node_stream_directions = {}  # (node_name, stream) -> direction
        for node in flat_nodes:
            node_name = node['name']
            for out_stream in node.get('output_streams', []):
                stream_info = self.parser.parse_stream_desc(out_stream)
                stream_name = stream_info['name']
                output_stream_to_node[(stream_name, node_name)] = (node_name, out_stream)
                node_stream = (node_name, out_stream)
                if node_stream not in self.nodes:
                    self.nodes.append(node_stream)
                node_stream_directions[node_stream] = "output"
        for node in flat_nodes:
            node_name = node['name']
            for in_stream in node.get('input_streams', []):
                node_stream = (node_name, in_stream)
                if node_stream not in self.nodes:
                    self.nodes.append(node_stream)
                node_stream_directions[node_stream] = "input"

        # Process input streams and build direct feed relationships
        for node in flat_nodes:
            node_name = node['name']

            # Create entries for input streams
            for in_stream in node.get('input_streams', []):
                node_stream = (node_name, in_stream)
                if node_stream not in self.nodes:
                    self.nodes.append(node_stream)

                stream_info = self.parser.parse_stream_desc(in_stream)
                in_stream_name = stream_info['name']

                # Find all matching output streams using streams_match
                for (stream_name, producer_name), producer_info in output_stream_to_node.items():
                    producer_stream = producer_info[1]

                    if self.parser.streams_match(in_stream, producer_stream):
                        # Add direct feed relationship
                        producer_node_stream = producer_info
                        consumer_node_stream = (node_name, in_stream)

                        # Update feeds mapping
                        if producer_node_stream not in self.feeds:
                            self.feeds[producer_node_stream] = []
                        if consumer_node_stream not in self.feeds[producer_node_stream]:
                            self.feeds[producer_node_stream].append(consumer_node_stream)

                        # Update feeds_from mapping
                        if consumer_node_stream not in self.feeds_from:
                            self.feeds_from[consumer_node_stream] = []
                        if producer_node_stream not in self.feeds_from[consumer_node_stream]:
                            self.feeds_from[consumer_node_stream].append(producer_node_stream)

        # Handle stream translations at graph boundaries
        self._process_graph_translations(flat_nodes)

        # Build the final graph representation
        flow_graph = {
            "nodes": [
                {"node": node[0], "stream": node[1], "stream-direction": node_stream_directions.get(node, "unknown")} for node in self.nodes
            ],
            "edges": []
        }

        # Add edges to the graph
        for source, targets in self.feeds.items():
            for target in targets:
                flow_graph["edges"].append({
                    "from": {"node": source[0], "stream": source[1]},
                    "to": {"node": target[0], "stream": target[1]}
                })

        return flow_graph

    def _flatten_nodes(self, pipeline_json: dict) -> List[dict]:
        """Recursively flatten the node hierarchy to get a list of all nodes with original names."""
        result = []

        # Add the current node
        result.append(pipeline_json)

        # Process child nodes
        for child in pipeline_json.get('nodes', []):
            result.extend(self._flatten_nodes(child))

        return result

    def _process_graph_translations(self, nodes: List[dict]) -> None:
        """
        Process stream translations at graph boundaries.

        This is crucial for connecting nodes inside graphs with nodes outside,
        accounting for stream name translations.
        """
        for graph_node in [node for node in nodes if node.get('type') == 'graph']:
            graph_name = graph_node['name']
            if 'graph_self_description' not in graph_node:
                continue
            gsd = graph_node['graph_self_description']
            # --- Outbound translation logic ---
            for out_stream in graph_node.get('output_streams', []):
                if '<outbound translation' in out_stream:
                    m = re.search(r'<outbound translation ([^ ]+) --> ([^>]+)>', out_stream)
                    if m:
                        internal_stream = m.group(1)
                        external_stream = m.group(2)
                        # Find internal producers of internal_stream
                        for node in self.nodes:
                            node_name, stream = node
                            if self.parser.streams_match(stream, internal_stream):
                                # Find external consumers that actually declare external_stream in their input_streams
                                for ext_node in nodes:
                                    ext_node_name = ext_node.get('name')
                                    if ext_node_name == graph_name:
                                        continue
                                    for ext_in_stream in ext_node.get('input_streams', []) or []:
                                        if self.parser.streams_match(ext_in_stream, external_stream):
                                            consumer = (ext_node_name, ext_in_stream)
                                            # Add direct feed relationship
                                            if node not in self.feeds:
                                                self.feeds[node] = []
                                            if consumer not in self.feeds[node]:
                                                self.feeds[node].append(consumer)
                                            if consumer not in self.feeds_from:
                                                self.feeds_from[consumer] = []
                                            if node not in self.feeds_from[consumer]:
                                                self.feeds_from[consumer].append(node)
            # Process input stream translations
            if 'input_streams' in gsd and 'input_streams' in graph_node:
                self._process_input_translations(
                    graph_name,
                    graph_node['input_streams'],
                    gsd['input_streams']
                )

            # Process output stream translations
            if 'output_streams' in gsd and 'output_streams' in graph_node:
                self._process_output_translations(
                    graph_name,
                    graph_node['output_streams'],
                    gsd['output_streams']
                )

    def _process_input_translations(self, graph_name: str, graph_inputs: List[str], self_inputs: List[str]) -> None:
        """Process input stream translations for a graph node."""
        # Match graph input streams with self-descriptive input streams
        for graph_input in graph_inputs:
            for self_input in self_inputs:
                if self.parser.streams_match(graph_input, self_input):
                    # Find nodes inside the graph that consume the translated stream
                    for node_stream in list(self.feeds_from.keys()):
                        node_name, stream = node_stream

                        # Connect graph boundary nodes directly
                        if self.parser.streams_match(stream, self_input):
                            # Find nodes outside that feed the graph
                            for producer_node_stream in self.feeds_from.get((graph_name, graph_input), []):
                                # Create direct feed relationship between outside producer and inside consumer
                                if producer_node_stream not in self.feeds_from[node_stream]:
                                    self.feeds_from[node_stream].append(producer_node_stream)

                                if node_stream not in self.feeds.get(producer_node_stream, []):
                                    if producer_node_stream not in self.feeds:
                                        self.feeds[producer_node_stream] = []
                                    self.feeds[producer_node_stream].append(node_stream)

    def _process_output_translations(self, graph_name: str, graph_outputs: List[str], self_outputs: List[str]) -> None:
        """Process output stream translations for a graph node."""
        # Match graph output streams with self-descriptive output streams
        for graph_output in graph_outputs:
            for self_output in self_outputs:
                if self.parser.streams_match(graph_output, self_output):
                    # Find nodes inside the graph that produce the translated stream
                    for node_stream, targets in list(self.feeds.items()):
                        node_name, stream = node_stream

                        # Connect graph boundary nodes directly
                        if self.parser.streams_match(stream, self_output):
                            # Find nodes outside that are fed by the graph
                            for target_node_stream in self.feeds.get((graph_name, graph_output), []):
                                # Create direct feed relationship between inside producer and outside consumer
                                if target_node_stream not in self.feeds.get(node_stream, []):
                                    if node_stream not in self.feeds:
                                        self.feeds[node_stream] = []
                                    self.feeds[node_stream].append(target_node_stream)

                                if node_stream not in self.feeds_from.get(target_node_stream, []):
                                    if target_node_stream not in self.feeds_from:
                                        self.feeds_from[target_node_stream] = []
                                    self.feeds_from[target_node_stream].append(node_stream)


def build_pipeline_flow(pipeline_parser, verbose_json_path, output_path):
    """
    Build and write the pipeline flow graph.

    Args:
        pipeline_parser: Instance of MediaPipePipelineParser
        verbose_json_path: Path to pipeline.verbose.json file
        output_path: Path to write the flow graph JSON
    """
    flow_builder = PipelineFlowBuilder(pipeline_parser)
    flow_graph = flow_builder.build_flow_graph(verbose_json_path)

    with open(output_path, 'w') as f:
        json.dump(flow_graph, f, indent=2)
