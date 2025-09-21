#!/usr/bin/env python3
"""
MediaPipe Complete Pipeline Mapper

This tool creates a comprehensive mapping of the complete MediaPipe hand inference
pipeline by recursively analyzing all subgraphs and finding all C++ implementations.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import json

from pipeline_parser import MediaPipePipelineParser, MediaPipeGraph
from call_graph_analyzer import MediaPipeCallGraphAnalyzer

@dataclass
class PipelineComponent:
    """Represents a component in the pipeline (calculator or subgraph)."""
    name: str
    type: str  # 'calculator', 'subgraph'
    definition_file: Optional[Path] = None
    implementation_files: List[Path] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    class_name: str = ""
    is_analyzed: bool = False

class CompletePipelineMapper:
    def __init__(self, mediapipe_source_path: Path):
        """Initialize the complete pipeline mapper."""
        self.mediapipe_source = Path(mediapipe_source_path)
        self.parser = MediaPipePipelineParser(mediapipe_source_path)
        
        # All discovered components
        self.components: Dict[str, PipelineComponent] = {}
        self.graph_files: Dict[str, Path] = {}
        self.analyzed_graphs: Set[str] = set()
        
        # Initialize by finding all graph files
        self._discover_all_graph_files()
    
    def _discover_all_graph_files(self):
        """Discover all .pbtxt graph definition files."""
        print("Discovering all MediaPipe graph files...")
        
        for root, dirs, files in os.walk(self.mediapipe_source):
            for file in files:
                if file.endswith('.pbtxt'):
                    path = Path(root) / file
                    graph_name = file.replace('.pbtxt', '')
                    self.graph_files[graph_name] = path
        
        print(f"Found {len(self.graph_files)} graph definition files")
    
    def map_complete_hand_pipeline(self) -> Dict:
        """Map the complete hand inference pipeline recursively."""
        print("Starting complete hand pipeline mapping...")
        
        # Start with the hand landmark tracking CPU graph
        self._analyze_graph_recursive('hand_landmark_tracking_cpu')
        
        # Also analyze the image wrapper version
        self._analyze_graph_recursive('hand_landmark_tracking_cpu_image') 
        
        # Find and analyze all referenced subgraphs
        self._find_and_analyze_all_subgraphs()
        
        # Analyze C++ implementations for all components
        self._analyze_all_implementations()
        
        return {
            'components': self.components,
            'total_components': len(self.components),
            'analyzed_graphs': list(self.analyzed_graphs),
            'graph_files': {name: str(path) for name, path in self.graph_files.items()},
            'mediapipe_source_path': str(self.mediapipe_source)
        }
    
    def _analyze_graph_recursive(self, graph_name: str, depth: int = 0) -> Optional[MediaPipeGraph]:
        """Recursively analyze a graph and all its subgraphs."""
        if depth > 10:  # Prevent infinite recursion
            print(f"  {'  ' * depth}Max depth reached for {graph_name}")
            return None
        
        if graph_name in self.analyzed_graphs:
            print(f"  {'  ' * depth}Already analyzed {graph_name}")
            return None
        
        # Find the graph file
        graph_path = self.graph_files.get(graph_name)
        if not graph_path:
            print(f"  {'  ' * depth}Graph file not found: {graph_name}")
            return None
        
        print(f"  {'  ' * depth}Analyzing graph: {graph_name}")
        self.analyzed_graphs.add(graph_name)
        
        try:
            # Parse the graph
            graph = self.parser.parse_pbtxt_file(graph_path)
            
            # Add all calculator nodes as components
            for node in graph.nodes:
                if node.calculator not in self.components:
                    self.components[node.calculator] = PipelineComponent(
                        name=node.calculator,
                        type='calculator'
                    )
            
            # Add all subgraphs as components and analyze them recursively
            for subgraph in graph.subgraphs:
                subgraph_name = subgraph.type.replace('Graph', '').lower()
                
                if subgraph.type not in self.components:
                    self.components[subgraph.type] = PipelineComponent(
                        name=subgraph.type,
                        type='subgraph',
                        definition_file=graph_path
                    )
                
                # Try to find and analyze the subgraph definition
                self._analyze_graph_recursive(subgraph_name, depth + 1)
            
            return graph
            
        except Exception as e:
            print(f"  {'  ' * depth}Error analyzing {graph_name}: {e}")
            return None
    
    def _find_and_analyze_all_subgraphs(self):
        """Find and analyze all subgraphs referenced in the pipeline."""
        print("Finding additional subgraphs...")
        
        # Look only at the hand-related graphs
        hand_related_graphs = [
            'palm_detection_cpu',
            'palm_detection_detection_to_roi',
            'hand_landmark_cpu', 
            'hand_landmark_landmarks_to_roi',
            'hand_landmark_landmarks_to_roi',
            'hand_landmark_detection',
            'hand_detection',
            'hand_detection_cpu'
        ]
        
        for graph_name in hand_related_graphs:
            if graph_name not in self.analyzed_graphs:
                self._analyze_graph_recursive(graph_name)
        
        # Also look for any graph files that contain "hand" in their name
        for graph_name, path in self.graph_files.items():
            if 'hand' in graph_name.lower() and graph_name not in self.analyzed_graphs:
                print(f"Found additional hand-related graph: {graph_name}")
                self._analyze_graph_recursive(graph_name)
    
    def _analyze_all_implementations(self):
        """Analyze C++ implementations for all discovered components."""
        print("Analyzing C++ implementations for all components...")
        
        calculator_sources = self._find_all_calculator_sources()
        
        for component_name, component in self.components.items():
            if component.type == 'calculator' and not component.is_analyzed:
                self._analyze_calculator_implementation(component, calculator_sources)
    
    def _find_all_calculator_sources(self) -> Dict[str, List[Path]]:
        """Find all calculator C++ source files."""
        print("Scanning for calculator source files...")
        
        calculator_files = {}
        
        for root, dirs, files in os.walk(self.mediapipe_source):
            for file in files:
                if file.endswith(('.cc', '.h', '.cpp', '.hpp')):
                    if 'calculator' in file.lower():
                        path = Path(root) / file
                        
                        # Extract calculator name from filename
                        base_name = file.lower()
                        for suffix in ['_calculator.cc', '_calculator.h', '_calculator.cpp', '_calculator.hpp']:
                            if base_name.endswith(suffix):
                                calc_name = base_name.replace(suffix, '')
                                calc_name_camel = self._snake_to_camel(calc_name) + 'Calculator'
                                
                                if calc_name_camel not in calculator_files:
                                    calculator_files[calc_name_camel] = []
                                calculator_files[calc_name_camel].append(path)
                                break
        
        print(f"Found {len(calculator_files)} calculator implementations")
        return calculator_files
    
    def _analyze_calculator_implementation(self, component: PipelineComponent, calculator_sources: Dict[str, List[Path]]):
        """Analyze the C++ implementation of a calculator."""
        calc_name = component.name
        
        # Try direct match first
        if calc_name in calculator_sources:
            component.implementation_files = calculator_sources[calc_name]
        else:
            # Try fuzzy matching
            for source_name, files in calculator_sources.items():
                if (calc_name.lower() in source_name.lower() or 
                    source_name.lower() in calc_name.lower() or
                    self._fuzzy_match(calc_name, source_name)):
                    component.implementation_files = files
                    break
        
        if component.implementation_files:
            print(f"  Found implementation for {calc_name}: {len(component.implementation_files)} files")
            
            # Analyze the source files
            for file_path in component.implementation_files:
                if file_path.suffix in ['.cc', '.cpp']:
                    file_analysis = self._analyze_source_file(file_path)
                    if file_analysis['class_name'] and not component.class_name:
                        component.class_name = file_analysis['class_name']
                    component.methods.extend(file_analysis['methods'])
                    component.dependencies.extend(file_analysis['dependencies'])
            
            # Remove duplicates
            component.methods = list(set(component.methods))
            component.dependencies = list(set(component.dependencies))
        else:
            print(f"  No implementation found for {calc_name}")
        
        component.is_analyzed = True
    
    def _analyze_source_file(self, file_path: Path) -> Dict:
        """Analyze a C++ source file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {'class_name': '', 'methods': [], 'dependencies': []}
        
        # Find class definitions
        class_pattern = r'class\s+(\w+)\s*(?::\s*public\s+[\w:]+)?\s*\{'
        class_matches = re.findall(class_pattern, content)
        class_name = class_matches[0] if class_matches else ""
        
        # Find method definitions  
        method_pattern = r'(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?\w+(?:\s*\*\s*|\s+)(\w+)\s*\([^)]*\)\s*(?:override\s*)?(?:const\s*)?[{;]'
        method_matches = re.findall(method_pattern, content)
        methods = [m for m in method_matches if not m.startswith('_') and len(m) > 2]
        
        # Find includes
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        includes = re.findall(include_pattern, content)
        mp_dependencies = [inc for inc in includes if 'mediapipe' in inc]
        
        return {
            'class_name': class_name,
            'methods': methods,
            'dependencies': mp_dependencies
        }
    
    def _snake_to_camel(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.capitalize() for x in components)
    
    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Check if two names are similar enough to be the same component."""
        name1_lower = name1.lower().replace('_', '').replace('calculator', '')
        name2_lower = name2.lower().replace('_', '').replace('calculator', '')
        
        return (name1_lower in name2_lower or 
                name2_lower in name1_lower or
                abs(len(name1_lower) - len(name2_lower)) <= 2)
    
    def generate_complete_report(self, mapping_results: Dict, output_file: Path):
        """Generate a comprehensive report of the complete pipeline mapping."""
        print(f"Generating complete pipeline report to {output_file}")
        
        report = {
            'analysis_info': {
                'mediapipe_version': 'v0.10.13',
                'total_components': mapping_results['total_components'],
                'analyzed_graphs': mapping_results['analyzed_graphs'],
                'source_path': mapping_results['mediapipe_source_path']
            },
            'components': {},
            'execution_flow': {
                'calculators': [],
                'subgraphs': []
            },
            'cpp_files': [],
            'graph_definitions': mapping_results['graph_files']
        }
        
        # Add component details
        for name, component in mapping_results['components'].items():
            report['components'][name] = {
                'type': component.type,
                'class_name': component.class_name,
                'implementation_files': [str(f) for f in component.implementation_files],
                'definition_file': str(component.definition_file) if component.definition_file else None,
                'methods': component.methods[:20],  # Limit for readability
                'dependencies': component.dependencies[:10],  # Limit for readability
                'is_analyzed': component.is_analyzed
            }
            
            # Collect by type
            if component.type == 'calculator':
                report['execution_flow']['calculators'].append(name)
            else:
                report['execution_flow']['subgraphs'].append(name)
            
            # Collect C++ files
            for file_path in component.implementation_files:
                if str(file_path) not in report['cpp_files']:
                    report['cpp_files'].append(str(file_path))
        
        # Write the report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
        
        # Generate markdown summary
        self._generate_complete_markdown_summary(mapping_results, output_file.with_suffix('.md'))
        
        print(f"Complete report written to {output_file}")
    
    def _generate_complete_markdown_summary(self, mapping_results: Dict, output_file: Path):
        """Generate a comprehensive markdown summary."""
        with open(output_file, 'w') as f:
            f.write("# Complete MediaPipe Hand Inference Pipeline C++ Call Graph\n\n")
            f.write("**What this document contains:**\n")
            f.write("This document provides a comprehensive analysis of the MediaPipe v0.10.13 Hand Landmark Tracking Pipeline, ")
            f.write("including all C++ calculator implementations, subgraph hierarchies, and complete source file mappings. ")
            f.write("It serves as a complete reference for understanding the entire pipeline structure and implementation details.\n\n")
            f.write("**Generated by:** `complete_pipeline_mapper.py`\n\n")
            f.write("---\n\n")
            
            # Summary stats
            total_components = mapping_results['total_components']
            calculators = [c for c in mapping_results['components'].values() if c.type == 'calculator']
            subgraphs = [c for c in mapping_results['components'].values() if c.type == 'subgraph']
            
            f.write("## Pipeline Overview\n\n")
            f.write(f"- **Total Components**: {total_components}\n")
            f.write(f"- **Calculators**: {len(calculators)}\n")
            f.write(f"- **Subgraphs**: {len(subgraphs)}\n")
            f.write(f"- **Analyzed Graphs**: {len(mapping_results['analyzed_graphs'])}\n\n")
            
            # List analyzed graphs
            f.write("## Analyzed Graph Definitions\n\n")
            for graph_name in sorted(mapping_results['analyzed_graphs']):
                f.write(f"- `{graph_name}.pbtxt`\n")
            f.write("\n")
            
            # Subgraph hierarchy
            f.write("## Subgraph Hierarchy\n\n")
            for component in sorted(subgraphs, key=lambda x: x.name):
                f.write(f"### {component.name}\n\n")
                if component.definition_file:
                    f.write(f"**Definition**: `{component.definition_file}`\n\n")
            
            # Calculator implementations  
            f.write("## Calculator Implementations\n\n")
            for component in sorted(calculators, key=lambda x: x.name):
                f.write(f"### {component.name}\n\n")
                f.write(f"- **Class**: `{component.class_name}`\n")
                f.write(f"- **Implementation Files**: {len(component.implementation_files)}\n")
                f.write(f"- **Methods Found**: {len(component.methods)}\n")
                f.write(f"- **Dependencies**: {len(component.dependencies)}\n\n")
                
                if component.implementation_files:
                    f.write("**C++ Files:**\n")
                    for impl_file in component.implementation_files:
                        f.write(f"- `{impl_file}`\n")
                    f.write("\n")
                
                if component.methods:
                    f.write("**Key Methods:**\n")
                    for method in component.methods[:10]:
                        f.write(f"- `{method}()`\n")
                    if len(component.methods) > 10:
                        f.write(f"- ... and {len(component.methods) - 10} more\n")
                    f.write("\n")
            
            # All C++ files involved
            all_cpp_files = []
            for component in mapping_results['components'].values():
                all_cpp_files.extend([str(f) for f in component.implementation_files])
            
            f.write("## Complete C++ File List\n\n")
            f.write(f"**Total C++ files involved in hand inference pipeline: {len(set(all_cpp_files))}**\n\n")
            
            for cpp_file in sorted(set(all_cpp_files)):
                f.write(f"- `{cpp_file}`\n")
            
        print(f"Complete markdown summary written to {output_file}")

def main():
    """Map the complete MediaPipe hand inference pipeline."""
    # Download MediaPipe source
    source_path = Path('mediapipe').resolve()
    

    # Map the complete pipeline
    mapper = CompletePipelineMapper(source_path)
    results = mapper.map_complete_hand_pipeline()

    # Generate comprehensive report
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / "all_calculators_and_cpp_sources.json"
    mapper.generate_complete_report(results, report_file)

    print(f"\n=== Complete Pipeline Mapping Complete ===")
    print(f"Total components found: {results['total_components']}")
    print(f"Analyzers graphs: {len(results['analyzed_graphs'])}")

    calculators = [c for c in results['components'].values() if c.type == 'calculator']
    subgraphs = [c for c in results['components'].values() if c.type == 'subgraph']

    print(f"Calculators: {len(calculators)}")
    print(f"Subgraphs: {len(subgraphs)}")

    # Count C++ files
    all_cpp_files = set()
    for component in results['components'].values():
        all_cpp_files.update(str(f) for f in component.implementation_files)

    print(f"Total C++ files involved: {len(all_cpp_files)}")
    print(f"Reports saved to: {output_dir}")
        

if __name__ == "__main__":
    main()