import React, { useState, useEffect } from 'react';

const CustomDataModal = ({ isOpen, onClose, onSubmit, algoType, currentAlgorithm }) => {
    const [inputValue, setInputValue] = useState('');
    const [error, setError] = useState('');

    useEffect(() => {
        if (isOpen) {
            const examples = {
                array: {
                    twoPointers: "5, 10, 15, 20, 25",
                    slidingWindow: "1, 2, 3, 4, 5",
                    kadanes: "-2, 1, -3, 4, -1, 2, 1",
                    default: "5, 10, 15, 20, 25"
                },
                tree: {
                    bstTraversal: "5\n3\n7\n1\n9"
                },
                graph: {
                    bfs: "A-B:4\nA-C:2",
                    dijkstra: "A-B:4\nA-C:2",
                    topologicalSort: "A-B:4\nA-C:2",
                    default: "A-B:4\nA-C:2"
                },
                'linked-list': {
                    linkedListReversal: "10, 20, 30, 40",
                    floydsCycle: "10, 20, 30, 40",
                    default: "10, 20, 30, 40"
                },
                board: {
                    backtracking: ". . Q\n. Q .\nQ . ."
                },
                other: {
                    bitManipulation: "170"
                },
                dp: {
                    dpFib: "0, 1, 1, 2, 3, 5, 8, 13, 21, 34"
                }
            };
            const example = examples[algoType]?.[currentAlgorithm] || examples[algoType]?.default || '';
            setInputValue(example);
            setError('');
        }
    }, [isOpen, algoType, currentAlgorithm]);

    if (!isOpen) return null;

    const placeholders = {
        array: "e.g., 5, 10, 15, 20, 25 (comma-separated numbers)",
        tree: "e.g., 5\n3\n7\n1\n9 (one number per line for BST)",
        graph: "e.g., A-B:4\nA-C:2 (one edge per line, e.g., 'A-B:4' for A to B with weight 4)",
        'linked-list': "e.g., 10, 20, 30, 40 (comma-separated numbers)",
        board: "e.g., . . Q\n. Q .\nQ . . (space-separated '.' or 'Q' per row)",
        other: "e.g., 170 (single integer)",
        dp: "e.g., 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 (10 comma-separated numbers)"
    };

    const handleSubmit = () => {
        try {
            onSubmit(inputValue);
            onClose();
            setError('');
        } catch (e) {
            setError(e.message);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
            <div className="bg-white rounded-lg shadow-2xl p-6 w-full max-w-lg">
                <h2 className="text-2xl font-bold mb-4">Input Custom Data</h2>
                <textarea
                    className="w-full h-40 p-2 border rounded font-mono text-sm"
                    placeholder={placeholders[algoType] || "Enter your data here..."}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                />
                {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
                <div className="flex justify-end space-x-4 mt-4">
                    <button onClick={onClose} className="px-4 py-2 rounded bg-gray-200 hover:bg-gray-300">Cancel</button>
                    <button onClick={handleSubmit} className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700">Visualize</button>
                </div>
            </div>
        </div>
    );
};

export default CustomDataModal;