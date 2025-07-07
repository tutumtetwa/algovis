// src/ArrayVisualizer.js
import React from 'react';

// This component visualizes an array as a series of bars.
// It accepts an 'array' of numbers and a 'highlights' object
// to color specific bars based on the algorithm's state.
const ArrayVisualizer = ({ array, highlights = {} }) => {
    if (!array) return null;

    return (
        <div className="flex justify-center items-end h-80 bg-gray-50 border p-2 rounded-lg space-x-1 w-full">
            {array.map((value, idx) => (
                <div key={idx} className="flex-1 text-center transition-all duration-300" style={{ height: `${value}%` }}>
                     <div
                        // The color of the bar is determined by the highlights object,
                        // falling back to a default blue.
                        className={`h-full w-full rounded-t-md ${highlights[idx]?.color || 'bg-blue-500'}`}
                     ></div>
                     <span className="text-xs font-bold">{value}</span>
                </div>
            ))}
        </div>
    );
}

export default ArrayVisualizer;
