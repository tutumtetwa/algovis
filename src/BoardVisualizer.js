// src/BoardVisualizer.js
import React from 'react';

const BoardVisualizer = ({ board, highlights }) => {
    if (!board) return null;
    const n = board.length;

    return (
        <div className="flex justify-center items-center p-4">
            <div className="grid border-gray-800 border-2" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
                {board.map((row, r_idx) => 
                    row.map((cell, c_idx) => {
                        const isLightSquare = (r_idx + c_idx) % 2 === 0;
                        let bgColor = isLightSquare ? 'bg-gray-200' : 'bg-gray-400';
                        if (highlights && highlights[`${r_idx}-${c_idx}`]) {
                            bgColor = highlights[`${r_idx}-${c_idx}`];
                        }

                        return (
                            <div key={`${r_idx}-${c_idx}`} className={`w-12 h-12 md:w-16 md:h-16 flex justify-center items-center transition-colors duration-300 ${bgColor}`}>
                                {cell === 'Q' && <span className="text-4xl md:text-5xl">♕</span>}
                            </div>
                        )
                    })
                )}
            </div>
        </div>
    );
};

export default BoardVisualizer;