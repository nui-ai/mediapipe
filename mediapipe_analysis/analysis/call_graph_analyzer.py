#!/usr/bin/env python3
"""
MediaPipe C++ Call Graph Analyzer

Analyzes the complete C++ call graph for MediaPipe pipelines by:
1. Parsing the graph definitions (.pbtxt files)
2. Mapping calculators to their C++ implementations  
3. Recursively analyzing sub-graphs and dependencies
4. Generating comprehensive call graph documentation
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import json
from collections import defaultdict

from pipeline_parser import MediaPipePipelineParser, GraphNode, SubGraph, MediaPipeGraph

@dataclass 
class CalculatorInfo:
    """Detailed information about a calculator implementation."""
    name: str
    header_files: List[Path] = field(default_factory=list)
    source_files: List[Path] = field(default_factory=list)
    class_name: str = ""
    methods: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    bazel_target: str = ""

@dataclass
class CallGraphNode:
    """Node in the complete call graph."""
    name: str
    type: str  # 'calculator', 'subgraph', 'function'
    implementation_files: List[Path] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    line_number: int = 0
    graph_file: Optional[Path] = None

class MediaPipeCallGraphAnalyzer:
    def __init__(self, mediapipe_source_path: Path):
        """
        Initialize the call graph analyzer.
        
        Args:
            mediapipe_source_path: Path to MediaPipe source directory
        """
        self.mediapipe_source = Path(mediapipe_source_path)
        self.parser = MediaPipePipelineParser(mediapipe_source_path)
        
        # Analysis results
        self.calculators: Dict[str, CalculatorInfo] = {}
        self.call_graph: Dict[str, CallGraphNode] = {}
        self.execution_order: List[str] = []
        
        # Cache for file analysis
        self._parsed_files: Set[Path] = set()
        self._bazel_targets: Dict[str, str] = {}
    
    def analyze_hand_inference_pipeline(self) -> Dict:
        """
        Analyze the complete hand inference pipeline call graph.
        
        Returns:
            Comprehensive analysis results including call graph and execution order
        """
        print("Starting comprehensive hand inference pipeline analysis...")
        
        # Get basic pipeline structure
        pipeline_analysis = self.parser.analyze_pipeline()
        main_graph = pipeline_analysis['main_graph']
        
        print(f"Analyzing pipeline with {len(main_graph.nodes)} nodes and {len(main_graph.subgraphs)} subgraphs")
        
        # Analyze all calculators in the pipeline
        self._analyze_calculators(main_graph.nodes)
        
        # Analyze all subgraphs recursively
        self._analyze_subgraphs(main_graph.subgraphs, pipeline_analysis['all_graph_files'])
        
        # Build the complete call graph
        self._build_call_graph(main_graph)
        
        # Determine execution order
        self._determine_execution_order(main_graph)
        
        # Find the actual hand landmark tracking CPU graph (not the image wrapper)
        hand_cpu_graph = self._find_and_analyze_hand_cpu_graph(pipeline_analysis['all_graph_files'])
        
        return {
            'main_graph': main_graph,
            'hand_cpu_graph': hand_cpu_graph,
            'calculators': self.calculators,
            'call_graph': self.call_graph,
            'execution_order': self.execution_order,
            'pipeline_files': pipeline_analysis,
            'mediapipe_source_path': self.mediapipe_source
        }
    
    def _find_and_analyze_hand_cpu_graph(self, all_graph_files: Dict[str, Path]) -> Optional[MediaPipeGraph]:
        """Find and analyze the actual hand landmark tracking CPU graph."""
        # Look for the main hand tracking CPU graph (not the image wrapper)
        for name, path in all_graph_files.items():
            if name == 'hand_landmark_tracking_cpu':
                print(f"Found main hand CPU graph: {path}")
                hand_cpu_graph = self.parser.parse_pbtxt_file(path)
                
                # Analyze this graph too
                self._analyze_calculators(hand_cpu_graph.nodes)
                self._analyze_subgraphs(hand_cpu_graph.subgraphs, all_graph_files)
                
                return hand_cpu_graph
        
        return None
    
    def _analyze_calculators(self, nodes: List[GraphNode]):
        """Analyze calculator implementations."""
        print(f"Analyzing {len(nodes)} calculator nodes...")
        
        for node in nodes:
            if node.calculator not in self.calculators:
                calc_info = self._analyze_single_calculator(node.calculator)
                if calc_info:
                    self.calculators[node.calculator] = calc_info
    
    def _analyze_single_calculator(self, calculator_name: str) -> Optional[CalculatorInfo]:
        """Analyze a single calculator's implementation."""
        print(f"Analyzing calculator: {calculator_name}")
        
        # Find calculator files
        header_files = []
        source_files = []
        
        # Search for calculator files with various naming patterns
        search_patterns = [
            f"*{self._snake_case(calculator_name)}*",
            f"*{calculator_name.lower()}*",
            f"*{calculator_name}*"
        ]
        
        for root, dirs, files in os.walk(self.mediapipe_source):
            for file in files:
                file_lower = file.lower()
                calc_lower = calculator_name.lower()
                
                # Check if this file is related to our calculator
                if (calc_lower in file_lower or 
                    self._snake_case(calculator_name) in file_lower or
                    file_lower.replace('_', '') == calc_lower.replace('_', '')):
                    
                    file_path = Path(root) / file
                    
                    if file.endswith('.h') or file.endswith('.hpp'):
                        header_files.append(file_path)
                    elif file.endswith('.cc') or file.endswith('.cpp'):
                        source_files.append(file_path)
        
        if not header_files and not source_files:
            print(f"  No implementation files found for {calculator_name}")
            return None
        
        # Analyze the source code
        class_name = ""
        methods = []
        dependencies = []
        
        for file_path in header_files + source_files:
            if file_path not in self._parsed_files:
                file_analysis = self._analyze_source_file(file_path)
                if file_analysis['class_name']:
                    class_name = file_analysis['class_name']
                methods.extend(file_analysis['methods'])
                dependencies.extend(file_analysis['dependencies'])
                self._parsed_files.add(file_path)
        
        # Find Bazel target
        bazel_target = self._find_bazel_target(calculator_name, source_files + header_files)
        
        calc_info = CalculatorInfo(
            name=calculator_name,
            header_files=header_files,
            source_files=source_files,
            class_name=class_name,
            methods=list(set(methods)),
            dependencies=list(set(dependencies)),
            bazel_target=bazel_target
        )
        
        print(f"  Found {len(header_files)} headers, {len(source_files)} sources")
        print(f"  Class: {class_name}, Methods: {len(methods)}")
        
        return calc_info
    
    def _analyze_source_file(self, file_path: Path) -> Dict:
        """Analyze a C++ source file for class definitions and methods."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
            return {'class_name': '', 'methods': [], 'dependencies': []}
        
        # Find class definitions
        class_pattern = r'class\s+(\w+)\s*(?::\s*public\s+[\w:]+)?\s*\{'
        class_matches = re.findall(class_pattern, content)
        class_name = class_matches[0] if class_matches else ""
        
        # Find method definitions
        method_pattern = r'(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?\w+(?:\s*\*\s*|\s+)(\w+)\s*\([^)]*\)\s*(?:override\s*)?(?:const\s*)?[{;]'
        method_matches = re.findall(method_pattern, content)
        
        # Filter out common false positives
        methods = [m for m in method_matches if not m.startswith('_') and len(m) > 2]
        
        # Find dependencies (#include statements)
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        includes = re.findall(include_pattern, content)
        
        # Find mediapipe-specific dependencies
        mp_dependencies = [inc for inc in includes if 'mediapipe' in inc]
        
        return {
            'class_name': class_name,
            'methods': methods,
            'dependencies': mp_dependencies
        }
    
    def _analyze_subgraphs(self, subgraphs: List[SubGraph], all_graph_files: Dict[str, Path]):
        """Recursively analyze subgraphs."""
        for subgraph in subgraphs:
            print(f"Analyzing subgraph: {subgraph.type}")
            
            # Find the graph definition file for this subgraph
            graph_file = None
            for name, path in all_graph_files.items():
                if subgraph.type.lower().replace('graph', '') in name.lower():
                    graph_file = path
                    break
            
            if graph_file:
                try:
                    subgraph_def = self.parser.parse_pbtxt_file(graph_file)
                    self._analyze_calculators(subgraph_def.nodes)
                    self._analyze_subgraphs(subgraph_def.subgraphs, all_graph_files)
                except Exception as e:
                    print(f"  Warning: Could not parse subgraph {graph_file}: {e}")
    
    def _build_call_graph(self, main_graph: MediaPipeGraph):
        """Build the complete call graph from analysis results."""
        print("Building call graph...")
        
        # Add calculator nodes
        for node in main_graph.nodes:
            calc_info = self.calculators.get(node.calculator)
            
            call_node = CallGraphNode(
                name=node.calculator,
                type='calculator',
                implementation_files=(calc_info.header_files + calc_info.source_files) if calc_info else [],
                line_number=node.line_number
            )
            
            self.call_graph[node.calculator] = call_node
        
        # Add subgraph nodes
        for subgraph in main_graph.subgraphs:
            call_node = CallGraphNode(
                name=subgraph.type,
                type='subgraph',
                line_number=subgraph.line_number
            )
            
            self.call_graph[subgraph.type] = call_node
        
        # Build connections based on stream flow
        self._build_stream_connections(main_graph)
    
    def _build_stream_connections(self, graph: MediaPipeGraph):
        """Build call connections based on stream flow."""
        # Map streams to their producers and consumers
        stream_producers = {}
        stream_consumers = defaultdict(list)
        
        # Process calculator nodes
        for node in graph.nodes:
            for output_stream in node.output_streams:
                stream_name = self._normalize_stream_name(output_stream)
                stream_producers[stream_name] = node.calculator
            
            for input_stream in node.input_streams:
                stream_name = self._normalize_stream_name(input_stream)
                stream_consumers[stream_name].append(node.calculator)
        
        # Process subgraphs
        for subgraph in graph.subgraphs:
            for output_stream in subgraph.output_streams:
                stream_name = self._normalize_stream_name(output_stream)
                stream_producers[stream_name] = subgraph.type
            
            for input_stream in subgraph.input_streams:
                stream_name = self._normalize_stream_name(input_stream)
                stream_consumers[stream_name].append(subgraph.type)
        
        # Build the call relationships
        for stream_name, producer in stream_producers.items():
            consumers = stream_consumers.get(stream_name, [])
            
            if producer in self.call_graph:
                self.call_graph[producer].calls.extend(consumers)
            
            for consumer in consumers:
                if consumer in self.call_graph:
                    self.call_graph[consumer].called_by.append(producer)
    
    def _determine_execution_order(self, graph: MediaPipeGraph):
        """Determine the execution order of the pipeline."""
        print("Determining execution order...")
        
        # Simple topological sort based on stream dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_name):
            if node_name in temp_visited:
                return  # Cycle detected, skip
            if node_name in visited:
                return
            
            temp_visited.add(node_name)
            
            if node_name in self.call_graph:
                for called in self.call_graph[node_name].calls:
                    visit(called)
            
            temp_visited.remove(node_name)
            visited.add(node_name)
            order.append(node_name)
        
        # Start with nodes that have no predecessors
        for node_name, node in self.call_graph.items():
            if not node.called_by:
                visit(node_name)
        
        # Add any remaining nodes
        for node_name in self.call_graph:
            if node_name not in visited:
                visit(node_name)
        
        self.execution_order = order
        print(f"Execution order determined: {len(order)} nodes")
    
    def _normalize_stream_name(self, stream: str) -> str:
        """Normalize stream name by removing tag prefixes."""
        if ':' in stream:
            return stream.split(':', 1)[1]
        return stream
    
    def _snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _find_bazel_target(self, calculator_name: str, files: List[Path]) -> str:
        """Find the Bazel target for a calculator."""
        if not files:
            return ""
        
        # Look for BUILD files in the same directory
        for file_path in files:
            build_file = file_path.parent / "BUILD"
            if build_file.exists():
                try:
                    with open(build_file, 'r') as f:
                        content = f.read()
                    
                    # Look for cc_library targets that might contain this calculator
                    target_pattern = r'cc_library\(\s*name\s*=\s*"([^"]+)"'
                    targets = re.findall(target_pattern, content)
                    
                    for target in targets:
                        if calculator_name.lower() in target.lower():
                            relative_path = file_path.parent.relative_to(self.mediapipe_source)
                            return f"//{relative_path}:{target}"
                except:
                    pass
        
        return ""
    
    def generate_report(self, analysis_results: Dict, output_file: Path):
        """Generate a comprehensive report of the call graph analysis."""
        print(f"Generating comprehensive report to {output_file}")
        
        report = {
            'analysis_timestamp': str(Path(__file__).stat().st_mtime),
            'mediapipe_version': 'v0.10.13',
            'pipeline_summary': {
                'main_graph_nodes': len(analysis_results['main_graph'].nodes),
                'main_graph_subgraphs': len(analysis_results['main_graph'].subgraphs),
                'total_calculators_analyzed': len(analysis_results['calculators']),
                'execution_order_length': len(analysis_results['execution_order'])
            },
            'calculators': {},
            'call_graph': {},
            'execution_order': analysis_results['execution_order'],
            'cpp_files_by_calculator': {},
            'bazel_targets': {}
        }
        
        # Add calculator details
        for name, calc_info in analysis_results['calculators'].items():
            report['calculators'][name] = {
                'class_name': calc_info.class_name,
                'header_files': [str(f) for f in calc_info.header_files],
                'source_files': [str(f) for f in calc_info.source_files],
                'methods': calc_info.methods,
                'dependencies': calc_info.dependencies,
                'bazel_target': calc_info.bazel_target
            }
            
            # Collect C++ files
            all_files = calc_info.header_files + calc_info.source_files
            report['cpp_files_by_calculator'][name] = [str(f) for f in all_files]
            
            if calc_info.bazel_target:
                report['bazel_targets'][name] = calc_info.bazel_target
        
        # Add call graph
        for name, node in analysis_results['call_graph'].items():
            report['call_graph'][name] = {
                'type': node.type,
                'implementation_files': [str(f) for f in node.implementation_files],
                'called_by': node.called_by,
                'calls': node.calls,
                'line_number': node.line_number
            }
        
        # Write the report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
        
        print(f"Report written to {output_file}")
        
        # Also generate a human-readable summary
        summary_file = output_file.with_suffix('.md')
        self._generate_markdown_summary(analysis_results, summary_file)
    
    def _generate_markdown_summary(self, analysis_results: Dict, output_file: Path):
        """Generate a comprehensive markdown analysis similar to the reference format."""
        with open(output_file, 'w') as f:
            f.write("# MediaPipe Hand Tracking - Complete C++ Call Graph Analysis\n\n")
            f.write("**What this document contains:**\n")
            f.write("This document maps the complete C++ call graph executed by the MediaPipe hand tracking pipeline, ")
            f.write("analyzing each calculator's role and the data flow between them. It provides detailed information ")
            f.write("about the two-stage pipeline (palm detection and hand landmark detection) including input/output ")
            f.write("types, key operations, and coordinate transformations.\n\n")
            f.write("**Generated by:** `call_graph_analyzer.py`\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write("This document maps the complete C++ call graph executed by the MediaPipe hand tracking pipeline, ")
            f.write("analyzing each calculator's role and the data flow between them.\n\n")
            
            # High-Level Pipeline Flow
            f.write("## High-Level Pipeline Flow\n\n")
            f.write("```\n")
            f.write("Input Image → Palm Detection → Hand ROI Generation → Hand Landmark Detection → Output Processing\n")
            f.write("```\n\n")
            
            # Main Analysis Results
            f.write("## Analysis Summary\n\n")
            f.write(f"- **Total Calculators**: {len(analysis_results['calculators'])}\n")
            f.write(f"- **Pipeline Stages**: 2 (Palm Detection + Hand Landmark Detection)\n")
            f.write(f"- **Graph Processing Nodes**: {len(analysis_results['call_graph'])}\n")
            f.write(f"- **Execution Order Steps**: {len(analysis_results['execution_order'])}\n\n")
            
            # Detailed Call Graph - organize by pipeline stages
            f.write("## Detailed Call Graph\n\n")
            
            # Analyze the hand CPU graph for more detailed pipeline information
            hand_cpu_graph = analysis_results.get('hand_cpu_graph')
            if hand_cpu_graph:
                self._write_detailed_pipeline_stages(f, hand_cpu_graph, analysis_results)
            else:
                self._write_basic_pipeline_info(f, analysis_results)
            
            # Key C++ Classes and Data Types
            f.write("## Key C++ Classes and Data Types\n\n")
            f.write("### Core Data Structures:\n")
            f.write("- `ImageFrame`: Input/output image container\n")
            f.write("- `NormalizedRect`: Region of interest specification\n")
            f.write("- `Detection`: Palm detection result with bounding box and keypoints\n")
            f.write("- `NormalizedLandmarkList`: 2D landmarks normalized to image space\n")
            f.write("- `LandmarkList`: 3D world landmarks in metric coordinates\n")
            f.write("- `ClassificationList`: Handedness classification result\n\n")
            
            f.write("### Key Calculator Base Classes:\n")
            f.write("- `CalculatorBase`: Base class for all calculators\n")
            f.write("- `Node`: Graph node containing calculator instance\n")
            f.write("- `CalculatorGraph`: Overall pipeline executor\n")
            f.write("- `Packet`: Data container for inter-calculator communication\n\n")
            
            f.write("### TensorFlow Lite Integration:\n")
            f.write("- `TfLiteModelCalculator`: Model loading and management\n")
            f.write("- `InferenceCalculator`: Model inference execution\n")
            f.write("- `TfLiteCustomOpResolverCalculator`: Custom operation support\n\n")
            
            # Implementation Details
            f.write("## Calculator Implementation Details\n\n")
            self._write_calculator_details(f, analysis_results)
            
            f.write("\nThis call graph represents the complete execution path from input image to final hand landmarks, ")
            f.write("showing how MediaPipe's graph-based architecture coordinates the complex multi-stage hand tracking pipeline.")
        
        print(f"Comprehensive markdown analysis written to {output_file}")
    
    def _write_detailed_pipeline_stages(self, f, hand_cpu_graph, analysis_results):
        """Write detailed information about pipeline stages based on the hand CPU graph."""
        # Group calculators by likely function based on their names
        palm_detection_calcs = []
        hand_landmark_calcs = []
        control_flow_calcs = []
        other_calcs = []
        
        for node in hand_cpu_graph.nodes:
            name = node.calculator
            if any(keyword in name.lower() for keyword in ['palm', 'detection']):
                palm_detection_calcs.append(node)
            elif any(keyword in name.lower() for keyword in ['landmark', 'hand']):
                hand_landmark_calcs.append(node)
            elif any(keyword in name.lower() for keyword in ['gate', 'loop', 'association', 'previous']):
                control_flow_calcs.append(node)
            else:
                other_calcs.append(node)
        
        # Palm Detection Pipeline
        if palm_detection_calcs:
            f.write("### 1. Palm Detection Pipeline\n\n")
            f.write("**Input**: ImageFrame (input image)\n")
            f.write("**Output**: std::vector<Detection> (palm detections)\n\n")
            f.write("#### Calculator Execution Flow:\n\n")
            
            for i, node in enumerate(palm_detection_calcs, 1):
                self._write_calculator_stage_info(f, node, i, analysis_results)
        
        # Hand Landmark Detection Pipeline 
        if hand_landmark_calcs:
            f.write("### 2. Hand Landmark Detection Pipeline\n\n")
            f.write("**Input**: ImageFrame, NormalizedRect (hand ROI)\n")
            f.write("**Output**: NormalizedLandmarkList, LandmarkList, ClassificationList\n\n")
            f.write("#### Calculator Execution Flow:\n\n")
            
            for i, node in enumerate(hand_landmark_calcs, 1):
                self._write_calculator_stage_info(f, node, i, analysis_results)
        
        # Main Pipeline Orchestration
        if control_flow_calcs:
            f.write("### 3. Main Pipeline Orchestration\n\n")
            f.write("#### Key Control Flow Calculators:\n\n")
            
            for i, node in enumerate(control_flow_calcs, 1):
                self._write_calculator_stage_info(f, node, i, analysis_results)
        
        # Other Components
        if other_calcs:
            f.write("### 4. Supporting Components\n\n")
            f.write("#### Additional Pipeline Calculators:\n\n")
            
            for i, node in enumerate(other_calcs, 1):
                self._write_calculator_stage_info(f, node, i, analysis_results)
    
    def _write_basic_pipeline_info(self, f, analysis_results):
        """Write basic pipeline information when detailed graph is not available."""
        f.write("### Main Pipeline Components\n\n")
        f.write("#### Execution Order:\n\n")
        for i, node_name in enumerate(analysis_results['execution_order'], 1):
            f.write(f"{i}. **{node_name}**\n")
            
            # Add details if available
            if node_name in analysis_results['calculators']:
                calc_info = analysis_results['calculators'][node_name]
                f.write(f"   - **Purpose**: Main pipeline component\n")
                f.write(f"   - **Implementation**: {len(calc_info.source_files)} C++ files\n")
                if calc_info.methods:
                    f.write(f"   - **Key Methods**: {', '.join(calc_info.methods[:3])}\n")
                f.write("\n")
        
        f.write("\n")
    
    def _write_calculator_stage_info(self, f, node, index, analysis_results):
        """Write detailed information about a calculator stage."""
        calc_name = node.calculator
        f.write(f"{index}. **{calc_name}**\n")
        
        # Get implementation details if available
        if calc_name in analysis_results['calculators']:
            calc_info = analysis_results['calculators'][calc_name]
            
            if hasattr(node, 'input_stream') and node.input_stream:
                f.write(f"   - **Input**: {', '.join(node.input_stream)}\n")
            if hasattr(node, 'output_stream') and node.output_stream:
                f.write(f"   - **Output**: {', '.join(node.output_stream)}\n")
            
            f.write(f"   - **Purpose**: {self._infer_calculator_purpose(calc_name)}\n")
            
            if calc_info.source_files:
                f.write(f"   - **Implementation**: {len(calc_info.source_files)} C++ files\n")
            
            if calc_info.methods:
                key_methods = [m for m in calc_info.methods if any(keyword in m.lower() 
                              for keyword in ['process', 'open', 'close', 'calculate', 'convert'])][:3]
                if key_methods:
                    f.write(f"   - **Key Methods**: {', '.join(key_methods)}\n")
        else:
            f.write(f"   - **Purpose**: {self._infer_calculator_purpose(calc_name)}\n")
            f.write(f"   - **Implementation**: Subgraph or external component\n")
        
        f.write("\n")
    
    def _infer_calculator_purpose(self, calc_name):
        """Infer the purpose of a calculator based on its name."""
        name_lower = calc_name.lower()
        
        if 'palm' in name_lower and 'detection' in name_lower:
            return "Detects palm regions in input image using SSD neural network"
        elif 'hand' in name_lower and 'landmark' in name_lower:
            return "Extracts 21 hand landmarks from hand ROI using regression network"
        elif 'gate' in name_lower:
            return "Controls data flow through pipeline based on conditions"
        elif 'loop' in name_lower:
            if 'begin' in name_lower:
                return "Starts parallel processing loop for multiple detections"
            elif 'end' in name_lower:
                return "Collects results from parallel processing loop"
        elif 'association' in name_lower:
            return "Associates current detections with previous frame trackings"
        elif 'previous' in name_lower or 'loopback' in name_lower:
            return "Provides temporal feedback loop for tracking"
        elif 'image' in name_lower and 'tensor' in name_lower:
            return "Converts input image to tensor format for neural network"
        elif 'tensor' in name_lower and ('detection' in name_lower or 'landmark' in name_lower):
            return "Converts neural network tensor outputs to structured data"
        elif 'nms' in name_lower or 'suppression' in name_lower:
            return "Removes overlapping detections using non-maximum suppression"
        elif 'letterbox' in name_lower:
            return "Adjusts coordinates for letterboxing transformations"
        elif 'projection' in name_lower:
            return "Projects coordinates between different coordinate systems"
        elif 'rect' in name_lower and 'size' in name_lower:
            return "Checks if sufficient regions detected for processing"
        elif 'clip' in name_lower:
            return "Limits number of detections to maximum allowed"
        elif 'properties' in name_lower:
            return "Extracts image metadata and properties"
        elif 'flow' in name_lower and 'limiter' in name_lower:
            return "Controls processing rate and resource usage"
        else:
            return "Pipeline component (purpose inferred from context)"
    
    def _write_calculator_details(self, f, analysis_results):
        """Write detailed implementation information for each calculator."""
        for name, calc_info in analysis_results['calculators'].items():
            if calc_info.source_files or calc_info.header_files:  # Only include if we have implementation details
                f.write(f"### {name}\n\n")
                f.write(f"- **Class**: `{calc_info.class_name or 'Unknown'}`\n")
                f.write(f"- **Implementation Files**: {len(calc_info.header_files + calc_info.source_files)}\n")
                
                if calc_info.source_files:
                    f.write(f"- **Source Files**: {len(calc_info.source_files)}\n")
                    for source in calc_info.source_files[:3]:  # Show first 3
                        source_name = str(source).split('/')[-1]  # Just filename
                        f.write(f"  - `{source_name}`\n")
                    if len(calc_info.source_files) > 3:
                        f.write(f"  - ... and {len(calc_info.source_files) - 3} more\n")
                
                if calc_info.methods:
                    f.write(f"- **Key Methods**: {len(calc_info.methods)} total\n")
                    # Show most relevant methods first
                    relevant_methods = [m for m in calc_info.methods if any(keyword in m.lower() 
                                      for keyword in ['process', 'open', 'close', 'calculate', 'convert'])]
                    other_methods = [m for m in calc_info.methods if m not in relevant_methods]
                    
                    shown_methods = relevant_methods[:5] + other_methods[:5-len(relevant_methods[:5])]
                    for method in shown_methods:
                        f.write(f"  - `{method}()`\n")
                    
                    remaining = len(calc_info.methods) - len(shown_methods)
                    if remaining > 0:
                        f.write(f"  - ... and {remaining} more\n")
                
                f.write("\n")

def main():
    """Analyze the MediaPipe hand inference pipeline."""

    # Download MediaPipe source
    source_path = Path('mediapipe').resolve()
    

    # Analyze the pipeline
    analyzer = MediaPipeCallGraphAnalyzer(source_path)
    results = analyzer.analyze_hand_inference_pipeline()

    # Generate report
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / "pipeline_execution_order_and_operations.json"
    analyzer.generate_report(results, report_file)

    print(f"\n=== Analysis Complete ===")
    print(f"Found {len(results['calculators'])} calculators")
    print(f"Built call graph with {len(results['call_graph'])} nodes")
    print(f"Execution order: {len(results['execution_order'])} steps")
    print(f"Reports saved to: {output_dir}")
        

if __name__ == "__main__":
    main()