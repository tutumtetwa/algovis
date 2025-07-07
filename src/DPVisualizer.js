// src/DPVisualizer.js
import React from 'react';

// Renders a 1D or 2D table for dynamic programming problems.
// It highlights cells based on the algorithm's progress.
const DPVisualizer = ({ table, highlights = {} }) => {
    if (!table) return null;
    const is2D = Array.isArray(table[0]);

    if (is2D) {
        // Render 2D table
        return <div>2D DP Visualizer Coming Soon</div>;
    }

    // Render 1D table (for Fibonacci, etc.)
    return (
        <div className="flex justify-center p-4">
            <div className="flex gap-1">
                {table.map((val, idx) => (
                    <div key={idx} className="flex flex-col items-center">
                        <div className={`w-16 h-16 border-2 flex items-center justify-center font-bold text-2xl transition-all duration-300 rounded-lg ${highlights[idx]?.color || 'bg-gray-100 border-gray-300'}`}>
                            {val}
                        </div>
                        <div className="text-sm mt-1 font-mono">dp[{idx}]</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default DPVisualizer;
