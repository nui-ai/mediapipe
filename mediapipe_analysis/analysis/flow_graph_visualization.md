# Flow Visualization

A slightly interactive D3.js-based visualization of a MediaPipe pipeline's flow with mild interactive focusing features.

## Features:

#### Search and Focus
- **Hover over any node** to focus on its immediate environment
- equivalently, search box at the top for easy node navigation ― **paste or type node name in the top search box** to instantly zoom to that node's environment
- Same clique behavior as hover - shows only the target node and its immediate connections
- **Clear the search box** to restore the full graph view
- (click outside the search box to be able to use keyboard shortcuts again)

#### Copy Node Information 
- **Left click any node** → Copies the node name to clipboard
- **Ctrl+click any node** → Copies the source file path to clipboard

#### Real-time Edge Length Control
- **`+` key** → Increase edge lengths a bit 
- **`-` key** → Decrease edge lengths a bit
- a minimum limit prevents edges from becoming too short (50px minimum)
- limitation: these keys are not active when the cursor is in the search box or the search box has focus


## Example Launch

make sure to change directory to the script's path. then run it with e.g. the following argument values: 

```bash
cd mediapipe_analysis/analysis
python flow_graph_visualization.py --font_scale 0.5 --repulsion_factor 0.3 --alpha_decay 1
```

## Fuzzy Layout Affordances

Use `python visualize_flow_graph.py` while tweaking the following argument values, since a force layout cannot be automagically optimized:

| Argument             | Default | When to Increase                                                                                                                                      | When to Decrease                                                                               | Description                                   |
|----------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `--font_scale`       | 0.5     | Nodes are too small to read<br/>High-resolution displays<br/>Long node names                                                                          | Text overlaps or looks too large<br/>Low-resolution displays<br/>Short node names              | Controls text size as fraction of node radius |
| `--alpha_decay`      | 0.03    | Graph nodes stop moving too quickly after initialization or after moving them<br/>Want smoother animations<br/>Complex graphs need more settling time | Graph takes too long to settle<br/>Want snappier interactions<br/>Simple graphs settle quickly | Lower values = longer movement/animation      |
| `--repulsion_factor` | 2.8     | Nodes are too close together<br/>Graph looks cluttered<br/>Text overlaps between nodes                                                                | Nodes are too spread out<br/>Graph looks sparse<br/>Want more compact layout                   | Higher values = more node separation          |


### Known Issue
when using +/- to adjust edge lengths, the graph may not re-layout immediately when the graph display has reached its stable state. click and hold the center node, or drag a node slightly to make the layout exit its stable state, while pressing +/- to see the effect immediately, otherwise it can get very confusing when you later nudge the graph and all your +/- adjustments take effect at once in a radical way.

## Technical Details

- Built with D3.js v7 for smooth force-directed layout
- Responsive design that adapts to window resizing (?)
- Real-time layout adjustments without page reload

## Requirements

- Input files: `output/json/streams-flow.json` and `output/json/pipeline.verbose.json` which are created by running `pipeline_parser.py` first.

