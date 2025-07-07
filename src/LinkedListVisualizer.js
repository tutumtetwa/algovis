// src/LinkedListVisualizer.js
import React from 'react';

const LinkedListVisualizer = ({ head, highlights = {} }) => {
    const nodes = [];
    let current = head;
    // FIX: Add a set to detect cycles and prevent infinite loops
    const visited = new Set();

    while (current && !visited.has(current.id)) {
        visited.add(current.id);
        nodes.push(current);
        current = current.next;
    }

    if (nodes.length === 0) {
        return <div className="h-80 flex items-center justify-center text-gray-400">Empty List</div>;
    }

    return (
        <div className="flex items-center justify-center p-8 space-x-4 overflow-x-auto">
            {nodes.map((node, idx) => (
                <React.Fragment key={node.id}>
                    <div className="flex flex-col items-center">
                        <div className={`w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-lg border-4 transition-all duration-300 ${highlights[node.id]?.color || 'bg-blue-500 border-blue-700'}`}>
                            {node.val}
                        </div>
                        <div className="flex mt-2 space-x-2">
                            {highlights[node.id]?.pointers?.map(p => (
                                <span key={p.name} className={`px-2 py-1 text-xs font-bold text-white rounded ${p.color}`}>
                                    {p.name}
                                </span>
                            ))}
                        </div>
                    </div>
                    {node.next && (
                        <div className={`text-4xl font-mono transition-all duration-300 ${highlights[node.id]?.arrowColor || 'text-gray-400'}`}>
                            →
                        </div>
                    )}
                </React.Fragment>
            ))}
             <div className="flex flex-col items-center ml-4">
                <div className="w-16 h-16 flex items-center justify-center text-gray-400 font-bold">
                    {current ? `(Cycle to ${current.val})` : 'null'}
                </div>
                <div className="flex mt-2 space-x-2">
                    {highlights.null?.pointers?.map(p => (
                        <span key={p.name} className={`px-2 py-1 text-xs font-bold text-white rounded ${p.color}`}>
                            {p.name}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default LinkedListVisualizer;
