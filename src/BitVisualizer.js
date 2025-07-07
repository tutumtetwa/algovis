// src/BitVisualizer.js
import React from 'react';

const BitVisualizer = ({ number, explanation, highlights = {} }) => {
    const toPaddedBinary = (num) => {
        return (num >>> 0).toString(2).padStart(8, '0');
    };

    const binaryString = toPaddedBinary(number);

    return (
        <div className="flex flex-col items-center justify-center p-8 bg-gray-800 text-white rounded-lg font-mono">
            <div className="text-2xl mb-4 text-gray-400">{explanation}</div>
            <div className="text-6xl font-bold mb-6 text-cyan-400">{number}</div>
            <div className="flex space-x-1">
                {binaryString.split('').map((bit, index) => (
                    <div key={index} className={`w-12 h-16 flex items-center justify-center text-4xl rounded-md transition-all duration-300 ${highlights[7 - index] ? highlights[7 - index].color : 'bg-gray-700'}`}>
                        {bit}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default BitVisualizer;
