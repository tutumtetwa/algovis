// src/GraphVisualizer.js
import React from 'react';

// Renders a graph using SVG. It shows nodes, edges, and weights.
// The 'highlights' object controls the color of nodes and edges
// and can display extra information like shortest path distances.
const GraphVisualizer = ({ graph, highlights = {} }) => {
    if (!graph) return <div className="h-96 flex items-center justify-center text-gray-400">No graph data</div>;

    return (
        <svg viewBox="0 0 400 300" className="w-full h-96 border rounded-lg bg-gray-50">
            {/* Render Edges First */}
            {Object.entries(graph.edges).map(([fromNode, neighbors]) => 
                neighbors.map(edge => {
                    const fromPos = graph.nodes[fromNode];
                    const toPos = graph.nodes[edge.node];
                    // Check if this edge should be highlighted
                    const isHighlighted = highlights[fromNode]?.edgeTo === edge.node || highlights[edge.node]?.edgeTo === fromNode;
                    return (
                        <g key={`${fromNode}-${edge.node}`}>
                            <line
                                x1={fromPos.x} y1={fromPos.y}
                                x2={toPos.x} y2={toPos.y}
                                className={`transition-all duration-300 ${isHighlighted ? 'stroke-pink-500 stroke-[3]' : 'stroke-gray-300 stroke-[1]'}`}
                            />
                            <text x={(fromPos.x + toPos.x) / 2} y={(fromPos.y + toPos.y) / 2 - 3} className="text-[9px] fill-gray-700 font-sans font-semibold">{edge.weight}</text>
                        </g>
                    );
                })
            )}
            {/* Render Nodes on Top */}
            {Object.entries(graph.nodes).map(([id, pos]) => (
                <g key={id} transform={`translate(${pos.x}, ${pos.y})`}>
                    <circle r="14" className={`transition-all duration-300 stroke-2 ${highlights[id]?.color || 'fill-blue-500 stroke-blue-700'}`}></circle>
                    <text textAnchor="middle" dy="4" className="fill-white font-bold text-[11px] font-sans select-none">{id}</text>
                    {/* Display distance from source for Dijkstra's */}
                    <text textAnchor="middle" dy="28" className="fill-black font-bold text-sm font-sans">
                        {highlights[id]?.distance === Infinity ? '∞' : highlights[id]?.distance}
                    </text>
                </g>
            ))}
        </svg>
    );
};

export default GraphVisualizer;
