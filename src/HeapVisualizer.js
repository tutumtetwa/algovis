// src/HeapVisualizer.js
import React from 'react';

// A recursive component to render a node and its children
const HeapNode = ({ array, index, highlights }) => {
    if (index >= array.length) {
        return null;
    }

    const leftChildIndex = 2 * index + 1;
    const rightChildIndex = 2 * index + 2;

    return (
        <div className="flex flex-col items-center p-2">
            {/* Current Node */}
            <div className={`w-16 h-16 rounded-full flex items-center justify-center font-bold text-xl border-4 transition-all duration-300 ${highlights[index]?.color || 'bg-blue-500 border-blue-700'}`}>
                <span className="text-white">{array[index]}</span>
            </div>

            {/* Children Container */}
            <div className="flex justify-center mt-4">
                {/* Left Child */}
                {leftChildIndex < array.length && (
                    <div className="relative px-4">
                        <div className="absolute -top-4 left-1/2 -translate-x-1/2 h-4 w-0.5 bg-gray-300"></div>
                        <HeapNode array={array} index={leftChildIndex} highlights={highlights} />
                    </div>
                )}
                {/* Right Child */}
                {rightChildIndex < array.length && (
                    <div className="relative px-4">
                        <div className="absolute -top-4 left-1/2 -translate-x-1/2 h-4 w-0.5 bg-gray-300"></div>
                        <HeapNode array={array} index={rightChildIndex} highlights={highlights} />
                    </div>
                )}
            </div>
        </div>
    );
};

const HeapVisualizer = ({ heap, highlights = {} }) => {
    if (!heap || heap.length === 0) {
        return <div className="h-80 flex items-center justify-center text-gray-400">No heap data</div>;
    }

    return (
        <div className="p-4 overflow-x-auto w-full">
            <HeapNode array={heap} index={0} highlights={highlights} />
        </div>
    );
};

export default HeapVisualizer;
