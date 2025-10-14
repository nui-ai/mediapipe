# flow fisualization

A slightly interactive D3.js-based visualization of a MediaPipe pipeline's flow with mild interactive focusing features.

## Features:

#### Search and Focus
- **Hover over any node** to focus on its immediate environment
- equivalently, search box at the top for easy node navigation ― **paste or type node name** to instantly zoom to that node's environment
- Same clique behavior as hover - shows only the target node and its immediate connections
- **Clear the search box** to restore the full graph view

#### Copy Node Information 
- **Left click any node** → Copies the node name to clipboard
- **Ctrl+click any node** → Copies the source file path to clipboard

#### Real-time Edge Length Control
- **`+` key** → Increase edge lengths a bit
- **`-` key** → Decrease edge lengths a bit
- a minimum limit prevents edges from becoming too short (50px minimum)

## 2. Fuzzy Layout Control

Use `python visualize_flow_graph.py` with the following arguments, since a force layout cannot be automagically optimized:

| Argument             | Default | When to Increase                                                                                                                                      | When to Decrease                                                                               | Description                                   |
|----------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `--font_scale`       | 0.5     | Nodes are too small to read<br/>High-resolution displays<br/>Long node names                                                                          | Text overlaps or looks too large<br/>Low-resolution displays<br/>Short node names              | Controls text size as fraction of node radius |
| `--alpha_decay`      | 0.03    | Graph nodes stop moving too quickly after initialization or after moving them<br/>Want smoother animations<br/>Complex graphs need more settling time | Graph takes too long to settle<br/>Want snappier interactions<br/>Simple graphs settle quickly | Lower values = longer movement/animation      |
| `--repulsion_factor` | 2.8     | Nodes are too close together<br/>Graph looks cluttered<br/>Text overlaps between nodes                                                                | Nodes are too spread out<br/>Graph looks sparse<br/>Want more compact layout                   | Higher values = more node separation          |

### Example Usage

make sure to change directory to the script's path. then run it: 

```bash
cd mediapipe_analysis/analysis
python flow_graph_visualization.py --font_scale 0.5 --repulsion_factor 0.3 --alpha_decay 1
```

## Technical Details

- Built with D3.js v7 for smooth force-directed layout
- Responsive design that adapts to window resizing (?)
- Real-time layout adjustments without page reload

## Requirements

- Input files: `output/json/streams-flow.json` and `output/json/pipeline.verbose.json`

