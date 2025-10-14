import os
import json
from flow_graph_builder_other import FlowGraphBuilder
import re
from collections import defaultdict

def get_node_id(node):
    # Use (name, source) as a unique identifier for each node (ignore line number)
    return (node['name'], node.get('source'))

def get_all_nodes(nodes):
    for node in nodes:
        yield node
        if node.get('type') == 'graph' and 'nodes' in node:
            yield from get_all_nodes(node['nodes'])

def build_node_maps(nodes):
    id_to_node = {}
    name_to_ids = defaultdict(list)
    for node in get_all_nodes(nodes):
        node_id = get_node_id(node)
        id_to_node[node_id] = node
        name_to_ids[node['name']].append(node_id)
    return id_to_node, name_to_ids

def make_md_link(node_name, source):
    if source:
        abs_source = os.path.abspath(source) if not os.path.isabs(source) else source
        return f"[{node_name}]({abs_source})"
    else:
        return node_name

def main():
    json_path = os.path.join(os.path.dirname(__file__), 'output', 'json', 'pipeline.verbose.json')
    md_dir = os.path.join(os.path.dirname(__file__), 'output', 'markdown')
    md_path = os.path.join(md_dir, 'relations.md')
    os.makedirs(md_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)
    nodes = data.get('nodes', data)
    builder = FlowGraphBuilder(nodes)
    id_to_node, name_to_ids = build_node_maps(nodes)

    with open(md_path, 'w') as md:
        md.write('# Node Direct Feed Relations\n\n')
        for node_id in sorted(id_to_node.keys()):
            name, source = node_id
            header_link = make_md_link(name, source)
            md.write(f'## {header_link}\n')
            direct_feeds = builder.get_direct_feeds_from(name)
            if direct_feeds:
                for feed_name in sorted(direct_feeds):
                    feed_ids = name_to_ids.get(feed_name, [])
                    for feed_id in feed_ids:
                        feed_name2, feed_source = feed_id
                        md.write(f'- {make_md_link(feed_name2, feed_source)}\n')
            else:
                md.write('no graph nodes feed from this node (the pipeline consumer may feed from them)\n')
            md.write('\n')
    print(f"Execution path relations written.")

if __name__ == '__main__':
    main()
