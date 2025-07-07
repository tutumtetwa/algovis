// src/TrieVisualizer.js
import React from 'react';

// A recursive component to render a Trie node and its children.
const TrieNodeVisualizer = ({ node, highlights, id }) => {
    const children = Object.entries(node.children);
    const isHighlighted = highlights[id]?.color;

    return (
        <div className="flex flex-col items-center p-2">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center font-mono font-bold border-2 transition-all duration-300 ${isHighlighted || (node.isEndOfWord ? 'bg-emerald-200 border-emerald-500' : 'bg-gray-200 border-gray-400')}`}>
                {id === 'root' ? ' ' : id.split('-').pop()}
            </div>
            {children.length > 0 && (
                <div className="flex justify-center mt-4 pl-4 border-l-2 border-gray-300">
                    {children.map(([char, childNode], index) => (
                        <div key={char} className="relative flex flex-col items-center">
                            <div className="absolute -top-4 left-1/2 -translate-x-1/2 h-4 w-0.5 bg-gray-300"></div>
                            <TrieNodeVisualizer node={childNode} highlights={highlights} id={`${id}-${char}`} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};


// The main component for visualizing a Trie data structure.
const TrieVisualizer = ({ trie, highlights = {} }) => {
    if (!trie) return <div className="h-80 flex items-center justify-center text-gray-400">No Trie data</div>;

    return (
        <div className="p-4 overflow-x-auto">
            <TrieNodeVisualizer node={trie.root} highlights={highlights} id="root" />
        </div>
    );
};

export default TrieVisualizer;
