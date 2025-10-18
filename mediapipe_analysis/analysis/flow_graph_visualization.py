import json
import webbrowser
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Generate D3.js graph HTML for mediapipe streams.')
parser.add_argument('--font_scale', type=float, default=0.5, help='Font size as a fraction of node radius (default: 0.5)')
parser.add_argument('--alpha_decay', type=float, default=0.03, help='D3 simulation alpha decay (default: 0.03, lower = longer movement)')
parser.add_argument('--repulsion_factor', type=float, default=2.8, help='Repulsion factor for D3 force simulation (default: 2.8, lower = more compact graph)')
args = parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Paths
base_dir = Path(__file__).parent
streams_flow_path = base_dir / 'output/json/streams-flow.json'
pipeline_verbose_path = base_dir / 'output/json/pipeline.verbose.json'
template_path = base_dir / 'flow_graph_visualization_template.html'
output_html_path = base_dir / 'output/flow_graph.html'

# Load data
streams_flow = load_json(streams_flow_path)
pipeline_verbose = load_json(pipeline_verbose_path)

# Build node->source mapping
def build_node_source_map(verbose):
    node_map = {}
    for node in verbose.get('nodes', []):
        name = node.get('name')
        source = node.get('source')
        if name and source:
            node_map[name] = source
    return node_map

node_source_map = build_node_source_map(pipeline_verbose)

# Extract nodes and edges from streams-flow.json
nodes = set()
edges = []
for record in streams_flow.get('nodes', []):
    node = record.get('node')
    if node:
        nodes.add(node)

# Edges extraction (fix: use 'edges' key)
if 'edges' in streams_flow:
    edge_records = streams_flow['edges']
    for rel in edge_records:
        from_node = rel.get('from', {}).get('node')
        to_node = rel.get('to', {}).get('node')
        if from_node and to_node:
            edges.append((from_node, to_node))
            nodes.add(from_node)
            nodes.add(to_node)
else:
    # fallback: previous logic for 'relations' or manual extraction
    if 'relations' in streams_flow:
        relations = streams_flow['relations']
        for rel in relations:
            from_node = rel.get('from', {}).get('node')
            to_node = rel.get('to', {}).get('node')
            if from_node and to_node:
                edges.append((from_node, to_node))
                nodes.add(from_node)
                nodes.add(to_node)
    else:
        # manual extraction (should not be needed now)
        pass

# Prepare graph data for d3.js
node_list = list(nodes)
links = [
    {"source": src, "target": tgt}
    for src, tgt in edges
]

def node_url(node):
    src = node_source_map.get(node)
    if src:
        return f"file://{src}"
    return None

nodes_json = json.dumps([{'id': n, 'url': node_url(n)} for n in node_list])
links_json = json.dumps(links)

# Read HTML template and inject data
with open(template_path, 'r') as f:
    template = f.read()
html = template.format(nodes_json=nodes_json, links_json=links_json, font_scale=args.font_scale, alpha_decay=args.alpha_decay, repulsion_factor=args.repulsion_factor)

with open(output_html_path, 'w') as f:
    f.write(html)

# Launch in browser
webbrowser.open(f'file://{output_html_path}', new=2)

print(f"number of nodes: {len(node_list)}")
print(f"number of edges: {len(links)}")
print(f"first node: {node_list[:1]}")
print(f"note that nodes which appear in the input graph by the same name more than once, are acknoweledged by this script and visualization as separate nodes, e.g. Node_1, Node_2, Node_3 etc when a node Node appeared more than once in the input graph.")
print(f"output HTML file generated and launched in the default browser: {output_html_path}")
print(f"")
