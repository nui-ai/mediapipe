import re
from collections import defaultdict

class ExecutionGraphBuilder:
    STREAM_PATTERN = re.compile(r'^[^:]+:[^:]+$')
    NAME_PATTERN = re.compile(r'^[^:]+$')
    CLONE_PATTERN = re.compile(r'^CLONE:\d+:(.+)$')

    def __init__(self, nodes):
        # Recursively flatten all nodes, including those in nested 'graph' nodes
        self.nodes = list(self._collect_all_nodes(nodes))
        self.node_names = set(node['name'] for node in self.nodes)
        self.output_streams = {}  # node_name -> set of output streams (tag:name or name)
        self.stream_to_node = {}  # stream (tag:name or name) -> node_name
        self.graph = defaultdict(set)  # node_name -> set of node_names that directly feed from it
        self._build_graph()

    def _collect_all_nodes(self, nodes):
        for node in nodes:
            yield node
            if node.get('type') == 'graph' and 'nodes' in node:
                # Recursively yield all nodes in this subgraph
                yield from self._collect_all_nodes(node['nodes'])

    def _parse_streams(self, streams, node_name, stream_type):
        parsed = set()
        for s in streams:
            m = self.CLONE_PATTERN.match(s)
            if m:
                # Special handling for CLONE:number:name
                clone_name = m.group(1)
                print(f"info: {stream_type} stream {s} of node '{node_name}' is a CLONE stream. it will be treated as '{clone_name}' for wiring.")
                parsed.add(clone_name)
            elif self.STREAM_PATTERN.match(s):
                parsed.add(s)
            elif self.NAME_PATTERN.match(s):
                print(f"info: {stream_type} stream {s} of node '{node_name}' does not follow the TAG:name convention. it will be taken as name-only.")
                parsed.add(s)
            else:
                print(f"warning: {stream_type} stream {s} of node '{node_name}' is of unknown syntax and could not be incorporated in the graph building.")
                parsed.add(s)
        return parsed

    def _build_graph(self):
        # First, map output streams to nodes
        for node in self.nodes:
            name = node['name']
            outputs = node.get('output_streams', [])
            parsed_outputs = self._parse_streams(outputs, name, 'output')
            self.output_streams[name] = parsed_outputs
            for stream in parsed_outputs:
                self.stream_to_node[stream] = name
        # Now, for each node, check its input streams and build edges
        for node in self.nodes:
            name = node['name']
            inputs = node.get('input_streams', [])
            parsed_inputs = self._parse_streams(inputs, name, 'input')
            for input_stream in parsed_inputs:
                if ':' in input_stream:
                    # input_stream is tag:name
                    in_tag, in_name = input_stream.split(':', 1)
                    for out_node, out_streams in self.output_streams.items():
                        for out_stream in out_streams:
                            if ':' in out_stream:
                                out_tag, out_name = out_stream.split(':', 1)
                                if (out_tag == in_tag or out_name == in_name) and out_node != name:
                                    self.graph[out_node].add(name)
                            else:
                                # out_stream is name-only
                                if out_stream == in_tag or out_stream == in_name:
                                    if out_node != name:
                                        self.graph[out_node].add(name)
                else:
                    # input_stream is name-only, keep existing logic
                    for out_node, out_streams in self.output_streams.items():
                        for out_stream in out_streams:
                            if (':' in out_stream and out_stream.split(':', 1)[1] == input_stream) or (out_stream == input_stream):
                                if out_node != name:
                                    self.graph[out_node].add(name)

    def get_direct_feeds_from(self, node_name):
        """Return set of node names that directly feed from the given node."""
        return set(self.graph.get(node_name, []))

    def get_recursive_feeds_from(self, node_name):
        """Return set of node names that recursively feed from the given node (any path)."""
        visited = set()
        stack = [node_name]
        while stack:
            current = stack.pop()
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited
