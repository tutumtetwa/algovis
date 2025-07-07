import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { create } from 'zustand';
import { algorithms } from './algorithm-data';
import ArrayVisualizer from './ArrayVisualizer';
import BoardVisualizer from './BoardVisualizer';
import DPVisualizer from './DPVisualizer';
import LinkedListVisualizer from './LinkedListVisualizer';
import TrieVisualizer from './TrieVisualizer';
import BitVisualizer from './BitVisualizer';
import HeapVisualizer from './HeapVisualizer';
import CustomDataModal from './CustomDataModal';


// --- BST Visualizer Component ---


const BSTVisualizer = ({ trie, highlights }) => {
    const [nodePositions, setNodePositions] = useState({});

    const getTreeDepth = (node) => {
        if (!node) return 0;
        return 1 + Math.max(getTreeDepth(node.children.left), getTreeDepth(node.children.right));
    };

    const getNodeCount = (node) => {
        if (!node) return 0;
        return 1 + getNodeCount(node.children.left) + getNodeCount(node.children.right);
    };

    const totalNodes = useMemo(() => getNodeCount(trie.root), [trie]);
    const treeDepth = useMemo(() => getTreeDepth(trie.root), [trie]);

    useEffect(() => {
        const positions = {};
        let xOffset = 0;

        const assignPositions = (node, depth = 0) => {
            if (!node) return;

            assignPositions(node.children.left, depth + 1);

            positions[node.id] = {
                x: xOffset * 80 + 100, // horizontal spacing
                y: depth * 100 + 80,   // vertical spacing
            };
            xOffset++;

            assignPositions(node.children.right, depth + 1);
        };

        if (trie.root) {
            assignPositions(trie.root);
        }

        setNodePositions(positions);
    }, [trie]);

    const renderEdges = (node) => {
        if (!node || !nodePositions[node.id]) return null;
        const thisPos = nodePositions[node.id];
        const elements = [];

        if (node.children.left && nodePositions[node.children.left.id]) {
            const leftPos = nodePositions[node.children.left.id];
            elements.push(
                <line
                    key={`${node.id}-left`}
                    x1={thisPos.x}
                    y1={thisPos.y}
                    x2={leftPos.x}
                    y2={leftPos.y}
                    className="stroke-teal-300 stroke-2 transition-all duration-300"
                />
            );
            elements.push(...renderEdges(node.children.left));
        }

        if (node.children.right && nodePositions[node.children.right.id]) {
            const rightPos = nodePositions[node.children.right.id];
            elements.push(
                <line
                    key={`${node.id}-right`}
                    x1={thisPos.x}
                    y1={thisPos.y}
                    x2={rightPos.x}
                    y2={rightPos.y}
                    className="stroke-teal-300 stroke-2 transition-all duration-300"
                />
            );
            elements.push(...renderEdges(node.children.right));
        }

        return elements;
    };

    const renderNodes = (node) => {
        if (!node || !nodePositions[node.id]) return null;
        const { x, y } = nodePositions[node.id];
        const highlight = highlights[node.id] || {};
        const color = highlight.color || 'fill-teal-200';
        const textColor = highlight.textColor || 'text-white';

        return (
            <g key={node.id}>
                <circle
                    cx={x}
                    cy={y}
                    r={20}
                    className={`fill-current ${color} stroke-teal-400 stroke-2 transition-all duration-300 filter drop-shadow-[0_0_8px_rgba(20,184,166,0.5)]`}
                />
                <text
                    x={x}
                    y={y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className={`text-sm font-semibold ${textColor}`}
                    style={{ pointerEvents: 'none' }}
                >
                    {node.value}
                </text>
                {renderNodes(node.children.left)}
                {renderNodes(node.children.right)}
            </g>
        );
    };

    const width = Math.max(800, totalNodes * 100); // More horizontal padding
    const height = Math.max(500, treeDepth * 120 + 100); // More vertical spacing

    if (!Object.keys(nodePositions).length) {
        return (
            <svg className="w-full h-96 bg-gray-900 rounded-lg">
                <text x="50%" y="50%" textAnchor="middle" className="text-white">
                    Loading tree...
                </text>
            </svg>
        );
    }

    return (
        <svg width={width} height={height} className="max-w-full bg-gray-900 rounded-lg p-4 overflow-visible">
            {renderEdges(trie.root)}
            {renderNodes(trie.root)}
        </svg>
    );
};

const GraphVisualizer = ({ graph, highlights }) => {
    if (!graph || !graph.nodes || !graph.edges) return (
        <div className="text-white text-center">No graph data</div>
    );

    return (
        <svg width="600" height="400" className="max-w-full bg-gray-900 rounded-lg p-2">
            {Object.entries(graph.nodes).map(([node, { x, y }]) => {
                const highlight = highlights[node] || {};
                const color = highlight.color || 'fill-gray-400';
                const stroke = highlight.stroke || 'stroke-gray-600';
                return (
                    <g key={node}>
                        <circle
                            cx={x}
                            cy={y}
                            r={20}
                            className={`fill-current ${color} ${stroke} stroke-2 transition-all duration-300`}
                        />
                        <text
                            x={x}
                            y={y}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="text-sm font-semibold text-white"
                            style={{ pointerEvents: 'none' }}
                        >
                            {node}
                        </text>
                    </g>
                );
            })}
            {Object.entries(graph.edges).map(([from, edges]) =>
                edges.map((edge, idx) => {
                    const to = edge.node;
                    const { x: x1, y: y1 } = graph.nodes[from];
                    const { x: x2, y: y2 } = graph.nodes[to];
                    return (
                        <line
                            key={`${from}-${to}-${idx}`}
                            x1={x1}
                            y1={y1}
                            x2={x2}
                            y2={y2}
                            className="stroke-teal-300 stroke-2 transition-all duration-300"
                        />
                    );
                })
            )}
        </svg>
    );
};

// --- Zustand Store for Global State ---
const useAlgoStore = create((set, get) => ({
    array: [], board: null, graph: null, dpTable: null, linkedList: null, trie: null, bitNumber: 0, heap: [],
    currentAlgorithm: 'twoPointers',
    currentLanguage: 'python',
    animationSpeed: 8000,
    isAutomated: true,
    isVisualizing: false,
    isComplete: false,
    isPaused: false,
    explanationText: "Select an algorithm.",
    showModal: false,
    setData: (data) => set((state) => ({ ...state, ...data })),
    setCurrentAlgorithm: (algo) => {
        set({ isAutomated: true, isVisualizing: false, isComplete: false, isPaused: false, currentAlgorithm: algo });
        get().generateData();
    },
    setCurrentLanguage: (lang) => set({ currentLanguage: lang }),
    setAnimationSpeed: (speed) => set({ animationSpeed: speed }),
    setIsAutomated: (val) => set({ isAutomated: val }),
    setIsVisualizing: (status) => set((state) => ({ isVisualizing: status, isComplete: !status && state.isVisualizing })),
    setIsPaused: (status) => set({ isPaused: status }),
    setExplanationText: (text) => set({ explanationText: text }),
    toggleModal: () => set((state) => ({ showModal: !state.showModal })),
    getAlgorithmProps: () => algorithms[get().currentAlgorithm],
    generateData: () => {
        const { type } = get().getAlgorithmProps();
        const { currentAlgorithm } = get();
        let data = { array: [], board: null, graph: null, dpTable: null, linkedList: null, trie: null, bitNumber: 0, heap: [] };

        if (type === 'array') {
            const newArr = Array.from({ length: 15 }, () => Math.floor(Math.random() * 90) + 10);
            if (currentAlgorithm === 'twoPointers') newArr.sort((a, b) => a - b);
            data.array = newArr;
        } else if (type === 'graph') {
            data.graph = {
                nodes: { A: { x: 50, y: 150 }, B: { x: 150, y: 50 }, C: { x: 150, y: 250 }, D: { x: 250, y: 150 }, E: { x: 350, y: 150 } },
                edges: parseGraphEdges("A-B:4\nA-C:2\nB-D:5\nC-D:8\nD-E:6")
            };
        } else if (type === 'dp') {
            data.dpTable = Array(10).fill('?');
        } else if (type === 'board') {
            data.board = Array(5).fill(null).map(() => Array(5).fill('.'));
        } else if (type === 'linked-list') {
            let head = { id: 0, val: 10, next: null };
            let current = head;
            for (let i = 1; i < 5; i++) {
                current.next = { id: i, val: (i + 1) * 10, next: null };
                current = current.next;
            }
            if (currentAlgorithm === 'floydsCycle') current.next = head.next;
            data.linkedList = head;
        } else if (type === 'tree') {
            if (currentAlgorithm === 'heap') {
                data.heap = Array.from({ length: 7 }, () => Math.floor(Math.random() * 90) + 10);
            } else if (currentAlgorithm === 'trie') {
                data.trie = { root: { children: {}, isEndOfWord: false } };
            } else if (currentAlgorithm === 'bstTraversal') {
                data.trie = {
                    root: {
                        value: 50, id: 50,
                        children: {
                            left: { value: 30, id: 30, children: { left: { value: 20, id: 20, children: {} }, right: { value: 40, id: 40, children: {} } } },
                            right: { value: 70, id: 70, children: { left: { value: 60, id: 60, children: {} }, right: { value: 80, id: 80, children: {} } } }
                        }
                    }
                };
            }
        } else if (type === 'other') {
            data.bitNumber = 170;
        }
        set({ ...data, explanationText: "New data generated. Ready to start again." });
    }
}));

// --- Helper Function to Parse Graph Edges ---
const parseGraphEdges = (text) => {
    const edges = {};
    const lines = text.trim().split('\n');
    for (const line of lines) {
        const [from, toWeight] = line.split('-');
        const [to, weight] = toWeight.split(':');
        if (from && to && !isNaN(parseInt(weight))) {
            edges[from] = edges[from] || [];
            edges[from].push({ node: to, weight: parseInt(weight) });
        }
    }
    return edges;
};

// --- Main App Component ---
export default function App() {
    const store = useAlgoStore();
    const {
        array, board, graph, dpTable, linkedList, trie, bitNumber, heap,
        currentAlgorithm, isAutomated, animationSpeed, isVisualizing, isComplete, isPaused,
        explanationText, showModal, getAlgorithmProps, toggleModal
    } = store;

    const [highlights, setHighlights] = useState({});
    const [stepCount, setStepCount] = useState(0);
    const stepperRef = useRef(null);
    const stepsHistory = useRef([]);
    const currentStepIndex = useRef(-1);

    const runStep = useCallback((step) => {
        if (!step || step.done) {
            store.setIsVisualizing(false);
            setStepCount(0);
            return;
        }
        const { explanation, highlights: newHighlights, data } = step.value;
        if (explanation) store.setExplanationText(explanation);
        if (newHighlights) setHighlights(newHighlights);
        if (data) store.setData(data);
        setStepCount((prev) => prev + 1);
    }, [store]);

    const handleNext = useCallback(() => {
        if (!stepperRef.current) return;
        const step = stepperRef.current.next();
        if (!step.done) {
            stepsHistory.current.push(step);
            currentStepIndex.current++;
            runStep(step);
        } else {
            runStep({ done: true });
        }
    }, [runStep]);

    const handlePrev = useCallback(() => {
        if (currentStepIndex.current > 0) {
            currentStepIndex.current--;
            setStepCount((prev) => prev - 1);
            const step = stepsHistory.current[currentStepIndex.current];
            runStep(step);
        }
    }, [runStep]);

    const stopVisualization = useCallback(() => {
        stepperRef.current = null;
        stepsHistory.current = [];
        currentStepIndex.current = -1;
        setHighlights({});
        setStepCount(0);
        store.setIsVisualizing(false);
        store.setIsPaused(false);
        store.setExplanationText("Visualization stopped. Ready to start again.");
        store.generateData();
        store.setCurrentAlgorithm(currentAlgorithm);
    }, [store, currentAlgorithm]);

    const startVisualization = useCallback(() => {
        const { isImplemented } = getAlgorithmProps();
        if (!isImplemented) {
            store.setExplanationText("This visualization is not yet implemented.");
            return;
        }
        stepsHistory.current = [];
        currentStepIndex.current = -1;
        setStepCount(0);
        setHighlights({});

        const data = { array, board, graph, dpTable, linkedList, trie, bitNumber, heap };

        if (currentAlgorithm === 'kadanes') stepperRef.current = kadanesGenerator(data.array);
        else if (currentAlgorithm === 'backtracking') stepperRef.current = nQueensGenerator(5);
        else if (currentAlgorithm === 'bitManipulation') stepperRef.current = bitManipulationGenerator(data.bitNumber);
        else if (currentAlgorithm === 'linkedListReversal') stepperRef.current = linkedListReversalGenerator(data.linkedList);
        else if (currentAlgorithm === 'dijkstra') stepperRef.current = dijkstraGenerator(data.graph);
        else if (currentAlgorithm === 'dpFib') stepperRef.current = dpFibGenerator();
        else if (currentAlgorithm === 'heap') stepperRef.current = buildHeapGenerator(data.heap);
        else if (currentAlgorithm === 'trie') stepperRef.current = trieInsertGenerator(data.trie, "ALGO");
        else if (currentAlgorithm === 'bstTraversal') stepperRef.current = bstInorderTraversalGenerator(data.trie.root);
        else if (currentAlgorithm === 'twoPointers') stepperRef.current = twoPointersGenerator(data.array, 100);
        else if (currentAlgorithm === 'slidingWindow') stepperRef.current = slidingWindowGenerator(data.array, 4);
        else if (currentAlgorithm === 'bfs') stepperRef.current = bfsGenerator(data.graph);
        else if (currentAlgorithm === 'topologicalSort') stepperRef.current = topologicalSortGenerator(data.graph);
        else if (currentAlgorithm === 'unionFind') stepperRef.current = unionFindGenerator(7);
        else if (currentAlgorithm === 'floydsCycle') stepperRef.current = floydsCycleGenerator(data.linkedList);
        else {
            store.setExplanationText("This visualization is not yet implemented.");
            return;
        }
        store.setIsVisualizing(true);
        store.setIsPaused(false);
        handleNext();
    }, [currentAlgorithm, getAlgorithmProps, handleNext, store, array, board, graph, dpTable, linkedList, trie, bitNumber, heap]);

    useEffect(() => {
        store.generateData();
    }, [currentAlgorithm]);

    useEffect(() => {
        if (isAutomated && isVisualizing && !isComplete && !isPaused) {
            const timer = setTimeout(() => {
                handleNext();
            }, 12000 - animationSpeed);
            return () => clearTimeout(timer);
        }
    }, [isAutomated, isVisualizing, isComplete, isPaused, animationSpeed, handleNext]);

    // --- Generator Functions for Algorithms ---
    function* buildHeapGenerator(initialArray) {
        let arr = [...initialArray];
        const n = arr.length;
        yield { explanation: "Starting Build-Heap process on the array.", data: { heap: arr } };

        for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
            yield* heapifyDown(arr, n, i);
        }
        yield { explanation: "Min-Heap has been successfully built.", data: { heap: arr }, highlights: {} };
    }

    function* heapifyDown(arr, n, i) {
        let smallest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let currentHighlights = { [i]: { color: 'bg-pink-500 border-pink-700' } };
        yield {
            explanation: `Heapifying subtree rooted at index ${i} (value ${arr[i]}).`,
            highlights: currentHighlights,
            data: { heap: arr }
        };

        if (left < n) {
            currentHighlights[left] = { color: 'bg-yellow-400 border-yellow-600' };
            yield { explanation: `Comparing with left child at index ${left} (value ${arr[left]}).`, highlights: currentHighlights, data: { heap: arr } };
            if (arr[left] < arr[smallest]) smallest = left;
        }
        if (right < n) {
            currentHighlights[right] = { color: 'bg-yellow-400 border-yellow-600' };
            yield { explanation: `Comparing with right child at index ${right} (value ${arr[right]}).`, highlights: currentHighlights, data: { heap: arr } };
            if (arr[right] < arr[smallest]) smallest = right;
        }

        if (smallest !== i) {
            yield {
                explanation: `Node ${arr[i]} violates heap property. Swapping with smaller child ${arr[smallest]}.`,
                highlights: { ...currentHighlights, [smallest]: { color: 'bg-green-500 border-green-700' } },
                data: { heap: arr }
            };
            [arr[i], arr[smallest]] = [arr[smallest], arr[i]];
            yield { explanation: `Swap complete.`, highlights: {}, data: { heap: [...arr] } };
            yield* heapifyDown(arr, n, smallest);
        } else {
            yield { explanation: `Node ${arr[i]} satisfies the heap property.`, highlights: { [i]: { color: 'bg-green-500 border-green-700' } }, data: { heap: arr } };
        }
    }

    function* kadanesGenerator(arr) {
        let maxSoFar = -Infinity, maxEndingHere = 0, start = 0, end = 0, s = 0;
        for (let i = 0; i < arr.length; i++) {
            maxEndingHere += arr[i];
            let explanation = `Index ${i} (${arr[i]}): Current window sum is ${maxEndingHere}.`;
            let newHighlights = {};
            for (let j = s; j <= i; j++) newHighlights[j] = { color: 'bg-yellow-400' };
            for (let j = start; j <= end; j++) newHighlights[j] = { color: 'bg-green-500' };
            newHighlights[i] = { color: 'bg-pink-500' };
            yield { explanation, highlights: newHighlights, data: { array: arr } };
            if (maxSoFar < maxEndingHere) {
                maxSoFar = maxEndingHere;
                start = s;
                end = i;
                yield { explanation: `New max sum found: ${maxSoFar}.`, highlights: newHighlights, data: { array: arr } };
            }
            if (maxEndingHere < 0) {
                maxEndingHere = 0;
                s = i + 1;
                yield { explanation: `Current window sum is negative. Resetting.`, highlights: newHighlights, data: { array: arr } };
            }
        }
        yield { explanation: `Kadane's complete. Max subarray sum is ${maxSoFar}.` };
    }

    function* bitManipulationGenerator(initialNum) {
        let num = initialNum;
        yield { explanation: `Starting with number ${num}.`, data: { bitNumber: num } };
        let count = 0;
        while (num > 0) {
            const LSB = num & -num;
            const bitIndex = Math.log2(LSB);
            yield { explanation: `Found least significant bit at position ${bitIndex}.`, data: { bitNumber: num }, highlights: { [bitIndex]: { color: 'bg-yellow-400' } } };
            num &= num - 1;
            count++;
            yield { explanation: `Clearing the bit. New number is ${num}. Count is ${count}.`, data: { bitNumber: num }, highlights: {} };
        }
        yield { explanation: `Finished. Total set bits (Hamming Weight) is ${count}.`, data: { bitNumber: initialNum } };
    }

    function* linkedListReversalGenerator(head) {
        let current = JSON.parse(JSON.stringify(head));
        let prev = null;
        while (current) {
            let highlights = {
                [current.id]: { color: 'bg-pink-500 border-pink-700', pointers: [{ name: 'curr', color: 'bg-pink-600' }] },
                ...(prev && { [prev.id]: { color: 'bg-green-500 border-green-700', pointers: [{ name: 'prev', color: 'bg-green-600' }] } }),
                ...(prev === null && { null: { pointers: [{ name: 'prev', color: 'bg-green-600' }] } })
            };
            yield { explanation: `Current node is ${current.val}. 'prev' is ${prev ? prev.val : 'null'}.`, highlights, data: { linkedList: current } };
            let next = current.next;
            current.next = prev;
            highlights[current.id].arrowColor = 'text-red-500';
            yield { explanation: `Point ${current.val}'s next to ${prev ? prev.val : 'null'}.`, highlights, data: { linkedList: current } };
            prev = current;
            current = next;
        }
        yield { explanation: 'Reversal complete.', data: { linkedList: prev }, highlights: {} };
    }

    function* nQueensGenerator(n) {
        let board = Array(n).fill(null).map(() => Array(n).fill('.'));
        function* backtrack(row) {
            if (row === n) {
                yield { explanation: 'Solution Found!', data: { board: [...board.map((r) => [...r])] } };
                return true;
            }
            for (let col = 0; col < n; col++) {
                yield {
                    explanation: `Row ${row}: Trying column ${col}.`,
                    highlights: { [`${row}-${col}`]: { color: 'bg-yellow-300' } },
                    data: { board: [...board.map((r) => [...r])] }
                };
                const isSafe = (r, c) => {
                    for (let i = 0; i < r; i++) if (board[i][c] === 'Q') return false;
                    for (let i = r - 1, j = c - 1; i >= 0 && j >= 0; i--, j--) if (board[i][j] === 'Q') return false;
                    for (let i = r - 1, j = c + 1; i >= 0 && j < n; i--, j++) if (board[i][j] === 'Q') return false;
                    return true;
                };
                if (isSafe(row, col)) {
                    board[row][col] = 'Q';
                    yield {
                        explanation: `Placed queen at (${row}, ${col}). Recursing...`,
                        highlights: {},
                        data: { board: [...board.map((r) => [...r])] }
                    };
                    if (yield* backtrack(row + 1)) return true;
                    board[row][col] = '.';
                    yield {
                        explanation: `Backtracking from (${row}, ${col}).`,
                        highlights: {},
                        data: { board: [...board.map((r) => [...r])] }
                    };
                }
            }
            return false;
        }
        yield* backtrack(0);
    }

    function* dijkstraGenerator(graph) {
        const startNode = 'A';
        let distances = {};
        let pq = [{ node: startNode, priority: 0 }];
        let localHighlights = {};
        Object.keys(graph.nodes).forEach((node) => {
            distances[node] = Infinity;
            localHighlights[node] = { color: 'fill-gray-400 stroke-gray-600', distance: Infinity };
        });
        distances[startNode] = 0;
        localHighlights[startNode] = { color: 'fill-orange-500 stroke-orange-700', distance: 0 };
        yield { explanation: `Initializing. Start node ${startNode} is 0, others are infinity.`, highlights: { ...localHighlights }, data: { graph } };

        while (pq.length > 0) {
            pq.sort((a, b) => a.priority - b.priority);
            const { node: currentNode } = pq.shift();
            if (localHighlights[currentNode].color === 'fill-green-600 stroke-green-800') continue;
            localHighlights[currentNode].color = 'fill-pink-500 stroke-pink-700';
            yield { explanation: `Visiting node ${currentNode}.`, highlights: { ...localHighlights }, data: { graph } };

            for (const edge of graph.edges[currentNode] || []) {
                if (!edge || typeof edge !== 'object' || !('node' in edge)) continue;
                const { node: neighborNode, weight } = edge;
                localHighlights[neighborNode] = localHighlights[neighborNode] || { color: 'fill-gray-400 stroke-gray-600', distance: Infinity };
                localHighlights[neighborNode].edgeTo = currentNode;
                yield { explanation: `Checking path to neighbor ${neighborNode}.`, highlights: { ...localHighlights }, data: { graph } };
                const newDist = distances[currentNode] + weight;
                if (newDist < distances[neighborNode]) {
                    distances[neighborNode] = newDist;
                    localHighlights[neighborNode] = { ...localHighlights[neighborNode], distance: newDist, color: 'fill-orange-500 stroke-orange-700' };
                    pq.push({ node: neighborNode, priority: newDist });
                    yield {
                        explanation: `Found shorter path to ${neighborNode}. New distance: ${newDist}.`,
                        highlights: { ...localHighlights },
                        data: { graph }
                    };
                } else {
                    yield { explanation: `Path to ${neighborNode} via ${currentNode} is not shorter.`, highlights: { ...localHighlights }, data: { graph } };
                }
                delete localHighlights[neighborNode].edgeTo;
            }
            localHighlights[currentNode].color = 'fill-green-600 stroke-green-800';
            yield { explanation: `Finished with node ${currentNode}.`, highlights: { ...localHighlights }, data: { graph } };
        }
        yield { explanation: "Dijkstra's complete. Shortest paths found." };
    }

    function* dpFibGenerator() {
        const n = 10;
        let dp = Array(n).fill('?');
        yield { explanation: 'Initializing DP table.', data: { dpTable: [...dp] } };

        dp[0] = 0;
        yield {
            explanation: 'Base case: dp[0] = 0',
            highlights: { 0: { color: 'bg-green-200 border-green-500' } },
            data: { dpTable: [...dp] }
        };

        dp[1] = 1;
        yield {
            explanation: 'Base case: dp[1] = 1',
            highlights: { 0: { color: 'bg-green-200 border-green-500' }, 1: { color: 'bg-green-200 border-green-500' } },
            data: { dpTable: [...dp] }
        };

        for (let i = 2; i < n; i++) {
            yield {
                explanation: `Calculating dp[${i}] = dp[${i - 1}] + dp[${i - 2}]`,
                highlights: { [i - 1]: { color: 'bg-blue-200 border-blue-500' }, [i - 2]: { color: 'bg-blue-200 border-blue-500' } }
            };
            dp[i] = dp[i - 1] + dp[i - 2];
            yield {
                explanation: `dp[${i}] is ${dp[i]}.`,
                highlights: { [i]: { color: 'bg-green-200 border-green-500' } },
                data: { dpTable: [...dp] }
            };
        }
        yield { explanation: 'Fibonacci sequence calculation complete.' };
    }

    function* trieInsertGenerator(trie, word) {
        let node = trie.root;
        let path = 'root';
        yield { explanation: `Starting to insert word: "${word}"`, data: { trie: { ...trie } }, highlights: { [path]: { color: 'bg-yellow-400' } } };
        for (const char of word) {
            path += `-${char}`;
            if (!node.children[char]) {
                node.children[char] = { children: {}, isEndOfWord: false };
                yield {
                    explanation: `Character '${char}' not found. Creating new node.`,
                    data: { trie: { ...trie } },
                    highlights: { [path]: { color: 'bg-green-400' } }
                };
            }
            node = node.children[char];
            yield { explanation: `Moving to node for '${char}'.`, data: { trie: { ...trie } }, highlights: { [path]: { color: 'bg-yellow-400' } } };
        }
        node.isEndOfWord = true;
        yield {
            explanation: `Finished inserting "${word}". Marking node as end of word.`,
            data: { trie: { ...trie } },
            highlights: { [path]: { color: 'bg-emerald-500' } }
        };
    }

    function* bstInorderTraversalGenerator(root) {
        const stack = [];
        let current = root;
        let path = [];
        while (current || stack.length > 0) {
            while (current) {
                stack.push(current);
                yield {
                    explanation: `Going left from ${current.value}. Pushing ${current.value} to stack.`,
                    data: { trie: { root } },
                    highlights: { [current.id]: { color: 'bg-yellow-400 border-yellow-600' } }
                };
                current = current.children.left;
            }
            current = stack.pop();
            path.push(current.value);
            yield {
                explanation: `Visiting node ${current.value}. Path: ${path.join(', ')}`,
                data: { trie: { root } },
                highlights: { [current.id]: { color: 'text-red-500' } }
            };
            current = current.children.right;
        }
        yield { explanation: `In-order traversal complete. Final path: ${path.join(', ')}` };
    }

    function* twoPointersGenerator(arr, target) {
        let left = 0;
        let right = arr.length - 1;
        yield {
            explanation: `Starting Two Pointers. Left: ${left}, Right: ${right}. Target: ${target}.`,
            highlights: { [left]: { color: 'bg-blue-500' }, [right]: { color: 'bg-blue-500' } },
            data: { array: arr }
        };

        while (left < right) {
            const sum = arr[left] + arr[right];
            yield {
                explanation: `Sum of arr[${left}] (${arr[left]}) + arr[${right}] (${arr[right]}) is ${sum}.`,
                highlights: { [left]: { color: 'bg-yellow-400' }, [right]: { color: 'bg-yellow-400' } },
                data: { array: arr }
            };

            if (sum === target) {
                yield {
                    explanation: `Target found! Indices ${left} and ${right}.`,
                    highlights: { [left]: { color: 'bg-green-500' }, [right]: { color: 'bg-green-500' } },
                    data: { array: arr }
                };
                return;
            } else if (sum < target) {
                yield {
                    explanation: `Sum ${sum} is less than target ${target}. Moving left pointer.`,
                    highlights: { [left]: { color: 'bg-red-500' }, [right]: { color: 'bg-blue-500' } },
                    data: { array: arr }
                };
                left++;
            } else {
                yield {
                    explanation: `Sum ${sum} is greater than target ${target}. Moving right pointer.`,
                    highlights: { [left]: { color: 'bg-blue-500' }, [right]: { color: 'bg-red-500' } },
                    data: { array: arr }
                };
                right--;
            }
            yield {
                explanation: `New pointers. Left: ${left}, Right: ${right}.`,
                highlights: { [left]: { color: 'bg-blue-500' }, [right]: { color: 'bg-blue-500' } },
                data: { array: arr }
            };
        }
        yield { explanation: `No pair found that sums to the target.`, highlights: {}, data: { array: arr } };
    }

    function* slidingWindowGenerator(arr, k) {
        let left = 0, currentSum = 0, maxSum = -Infinity;
        yield { explanation: `Starting Sliding Window with size k=${k}.`, data: { array: arr } };

        for (let right = 0; right < arr.length; right++) {
            currentSum += arr[right];
            let highlights = {};
            for (let i = left; i <= right; i++) highlights[i] = { color: 'bg-yellow-400' };
            yield { explanation: `Window [${left}, ${right}]. Current sum: ${currentSum}.`, highlights, data: { array: arr } };

            if (right - left + 1 === k) {
                if (currentSum > maxSum) {
                    maxSum = currentSum;
                    yield { explanation: `New max sum found: ${maxSum}.`, highlights, data: { array: arr } };
                }
                currentSum -= arr[left];
                left++;
            }
        }
        yield { explanation: `Sliding Window complete. Max sum is ${maxSum}.`, highlights: {} };
    }

    function* bfsGenerator(graph) {
        const startNode = 'A';
        const queue = [startNode];
        const visited = new Set([startNode]);
        let highlights = { [startNode]: { color: 'fill-orange-500 stroke-orange-700' } };
        yield { explanation: `Starting BFS from node ${startNode}. Queue: [${queue.join(', ')}]`, highlights, data: { graph } };

        while (queue.length > 0) {
            const u = queue.shift();
            highlights[u] = { color: 'fill-pink-500 stroke-pink-700' };
            yield { explanation: `Visiting node ${u}. Queue: [${queue.join(', ')}]`, highlights, data: { graph } };

            const edges = graph.edges[u] || [];
            if (!Array.isArray(edges)) {
                yield { explanation: `No valid edges found for node ${u}. Skipping.`, highlights, data: { graph } };
                continue;
            }

            for (const edge of edges) {
                if (!edge || typeof edge !== 'object' || !('node' in edge)) {
                    yield { explanation: `Invalid edge format for node ${u}. Skipping edge.`, highlights, data: { graph } };
                    continue;
                }
                const v = edge.node;
                if (!visited.has(v)) {
                    visited.add(v);
                    queue.push(v);
                    highlights[v] = { color: 'fill-yellow-400 stroke-yellow-600' };
                    yield { explanation: `Found unvisited neighbor ${v}. Adding to queue.`, highlights, data: { graph } };
                }
            }
            highlights[u] = { color: 'fill-green-600 stroke-green-800' };
            yield { explanation: `Finished with node ${u}.`, highlights, data: { graph } };
        }
        yield { explanation: 'BFS complete.' };
    }

    function* topologicalSortGenerator(graph) {
        const adj = graph.edges;
        const nodes = Object.keys(graph.nodes);
        const inDegree = {};
        for (const node of nodes) inDegree[node] = 0;
        for (const u in adj) {
            for (const edge of adj[u] || []) {
                if (edge && typeof edge === 'object' && 'node' in edge) {
                    inDegree[edge.node]++;
                }
            }
        }

        const queue = nodes.filter((node) => inDegree[node] === 0);
        let result = [];
        let highlights = {};
        queue.forEach((node) => (highlights[node] = { color: 'fill-yellow-400 stroke-yellow-600' }));
        yield { explanation: `Initializing. Nodes with in-degree 0 are: ${queue.join(', ')}.`, highlights, data: { graph } };

        while (queue.length > 0) {
            const u = queue.shift();
            result.push(u);
            highlights[u] = { color: 'fill-green-600 stroke-green-800' };
            yield { explanation: `Processing node ${u}. Sorted order: ${result.join(' → ')}.`, highlights, data: { graph } };

            for (const edge of adj[u] || []) {
                if (edge && typeof edge === 'object' && 'node' in edge) {
                    const v = edge.node;
                    inDegree[v]--;
                    highlights[v] = { ...highlights[v], edgeTo: u };
                    yield { explanation: `Decrementing in-degree of neighbor ${v}.`, highlights, data: { graph } };
                    if (inDegree[v] === 0) {
                        queue.push(v);
                        highlights[v] = { color: 'fill-yellow-400 stroke-yellow-600' };
                        yield { explanation: `Node ${v} now has in-degree 0. Adding to queue.`, highlights, data: { graph } };
                    }
                    delete highlights[v].edgeTo;
                }
            }
        }
        if (result.length === nodes.length) {
            yield { explanation: `Topological sort complete. Final order: ${result.join(' → ')}.` };
        } else {
            yield { explanation: 'Graph has a cycle! Topological sort not possible.' };
        }
    }

    function* unionFindGenerator(size) {
        const parent = Array.from({ length: size }, (_, i) => i);
        const graph = { nodes: {}, edges: {} };
        for (let i = 0; i < size; i++) {
            graph.nodes[i] = { x: 50 + (i % 4) * 100, y: 75 + Math.floor(i / 4) * 150 };
            graph.edges[i] = [];
        }
        yield { explanation: 'Initializing Union-Find. Each node is its own parent.', data: { graph }, highlights: {} };

        const find = (i) => {
            if (parent[i] === i) return i;
            return find(parent[i]);
        };

        const union = function* (i, j) {
            const rootI = find(i);
            const rootJ = find(j);
            yield {
                explanation: `Union operation on ${i} and ${j}. Root of ${i} is ${rootI}, root of ${j} is ${rootJ}.`,
                data: { graph },
                highlights: { [i]: { color: 'fill-yellow-400' }, [j]: { color: 'fill-yellow-400' } }
            };
            if (rootI !== rootJ) {
                parent[rootJ] = rootI;
                graph.edges[i].push({ node: j, weight: '' });
                yield {
                    explanation: `Connecting ${j} to ${i}.`,
                    data: { graph: { ...graph } },
                    highlights: { [i]: { color: 'fill-green-500' }, [j]: { color: 'fill-green-500' } }
                };
            } else {
                yield {
                    explanation: `${i} and ${j} are already in the same set.`,
                    data: { graph },
                    highlights: { [i]: { color: 'fill-pink-500' }, [j]: { color: 'fill-pink-500' } }
                };
            }
        };

        yield* union(1, 2);
        yield* union(2, 3);
        yield* union(4, 5);
        yield* union(5, 6);
        yield* union(1, 4);
        yield { explanation: 'Union-Find operations complete.' };
    }

    function* floydsCycleGenerator(head) {
        if (!head || !head.next) {
            yield { explanation: 'No cycle possible in a list with less than 2 nodes.' };
            return;
        }
        let slow = head;
        let fast = head;
        yield {
            explanation: "Starting Floyd's Cycle Detection. Both pointers at head.",
            data: { linkedList: head },
            highlights: { [slow.id]: { pointers: [{ name: 'slow/fast', color: 'bg-purple-600' }] } }
        };

        while (fast && fast.next) {
            slow = slow.next;
            fast = fast.next.next;

            const slowPointer = { [slow.id]: { pointers: [{ name: 'slow', color: 'bg-blue-600' }] } };
            const fastPointer = fast ? { [fast.id]: { pointers: [{ name: 'fast', color: 'bg-red-600' }] } } : {};
            const combinedPointers = slow.id === fast?.id ? { [slow.id]: { pointers: [{ name: 'slow/fast', color: 'bg-purple-600' }] } } : { ...slowPointer, ...fastPointer };

            yield {
                explanation: `Slow moves to ${slow.val}. Fast moves to ${fast ? fast.val : 'null'}.`,
                data: { linkedList: head },
                highlights: combinedPointers
            };

            if (slow === fast) {
                yield {
                    explanation: `Cycle detected! Pointers met at node ${slow.val}.`,
                    data: { linkedList: head },
                    highlights: { [slow.id]: { color: 'bg-green-500 border-green-700', pointers: [{ name: 'MEET', color: 'bg-green-600' }] } }
                };
                return;
            }
        }
        yield { explanation: 'No cycle detected. Fast pointer reached null.' };
    }

    const handleCustomDataSubmit = (text) => {
    try {
        const { type } = getAlgorithmProps();
        let newData = {};
        if (type === 'array') {
            const parsed = text.split(',').map((n) => parseInt(n.trim())).filter((n) => !isNaN(n));
            if (parsed.length === 0) throw new Error('Invalid array format. Use comma-separated numbers (e.g., 5, 10, 15).');
            if (currentAlgorithm === 'twoPointers' || currentAlgorithm === 'slidingWindow' || currentAlgorithm === 'kadanes') {
                if (parsed.length < 2) throw new Error('Array must have at least 2 numbers.');
                if (currentAlgorithm === 'twoPointers') parsed.sort((a, b) => a - b);
            }
            newData.array = parsed;
            store.setExplanationText(`Custom array set with values: ${parsed.join(', ')}. Use the Start button to begin.`);
        } else if (type === 'tree' && currentAlgorithm === 'bstTraversal') {
            const values = text.split('\n').map((n) => parseInt(n.trim())).filter((n) => !isNaN(n) && n >= 0);
            if (values.length === 0) throw new Error('Invalid BST input. Use one number per line (e.g., 5\n3\n7).');
            const uniqueValues = [...new Set(values)]; // Remove duplicates
            if (uniqueValues.length < 1) throw new Error('BST requires at least one unique number.');
            const buildBST = (values) => {
                if (values.length === 0) return null;
                const mid = Math.floor(values.length / 2);
                const root = { value: values[mid], id: values[mid], children: { left: null, right: null } };
                root.children.left = buildBST(values.slice(0, mid));
                root.children.right = buildBST(values.slice(mid + 1));
                return root;
            };
            const bstRoot = { root: buildBST(uniqueValues.sort((a, b) => a - b)) };
            newData.trie = bstRoot;
            store.setExplanationText(`Custom BST set with values: ${uniqueValues.join(', ')}. Use the Start button to begin.`);
        } else if (type === 'tree' && currentAlgorithm === 'heap') {
            const values = text.split(',').map((n) => parseInt(n.trim())).filter((n) => !isNaN(n));
            if (values.length === 0) throw new Error('Invalid heap format. Use comma-separated numbers (e.g., 5, 10, 15).');
            if (values.length < 1) throw new Error('Heap requires at least one number.');
            newData.heap = values;
            store.setExplanationText(`Custom heap set with values: ${values.join(', ')}. Use the Start button to begin.`);
        } else if (type === 'graph') {
            const lines = text.trim().split('\n');
            const nodes = new Set();
            const edges = {};
            for (const line of lines) {
                const [from, toWeight] = line.split('-');
                const [to, weight] = toWeight.split(':');
                if (from && to && !isNaN(parseInt(weight))) {
                    nodes.add(from);
                    nodes.add(to);
                    edges[from] = edges[from] || [];
                    edges[from].push({ node: to, weight: parseInt(weight) });
                } else {
                    throw new Error('Invalid graph format. Use "A-B:4" per line (e.g., A-B:4\nA-C:2), with a valid weight.');
                }
            }
            if (nodes.size === 0) throw new Error('No nodes defined in graph.');
            newData.graph = { nodes: Object.fromEntries([...nodes].map(n => [n, { x: 50 + Math.random() * 300, y: 50 + Math.random() * 300 }])), edges };
            store.setExplanationText(`Custom graph set with ${nodes.size} nodes. Use the Start button to begin.`);
        } else if (type === 'linked-list') {
            const values = text.split(',').map((n) => parseInt(n.trim())).filter((n) => !isNaN(n));
            if (values.length === 0) throw new Error('Invalid linked list format. Use comma-separated numbers (e.g., 10, 20, 30).');
            if (values.length < 1) throw new Error('Linked list requires at least one number.');
            let head = { id: 0, val: values[0], next: null };
            let current = head;
            for (let i = 1; i < values.length; i++) {
                current.next = { id: i, val: values[i], next: null };
                current = current.next;
            }
            if (currentAlgorithm === 'floydsCycle' && values.length > 1) current.next = head.next;
            newData.linkedList = head;
            store.setExplanationText(`Custom linked list set with values: ${values.join(', ')}. Use the Start button to begin.`);
        } else if (type === 'board' && currentAlgorithm === 'backtracking') {
            const rows = text.trim().split('\n').map(row => row.trim().split(' ').filter(cell => cell === '.' || cell === 'Q'));
            if (rows.length === 0 || rows.some(row => row.length !== rows[0].length || row.length < 1)) {
                throw new Error('Invalid board format. Use rows of space-separated "." or "Q" (e.g., ". . Q\n. Q .\nQ . .").');
            }
            newData.board = rows;
            store.setExplanationText(`Custom board set with ${rows.length}x${rows[0].length} size. Use the Start button to begin.`);
        } else if (type === 'other' && currentAlgorithm === 'bitManipulation') {
            const num = parseInt(text.trim());
            if (isNaN(num)) throw new Error('Invalid number format. Enter a single integer (e.g., 170).');
            newData.bitNumber = num;
            store.setExplanationText(`Custom number set: ${num}. Use the Start button to begin.`);
        } else if (type === 'dp' && currentAlgorithm === 'dpFib') {
            const nums = text.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
            if (nums.length !== 10) throw new Error('DP table must have exactly 10 comma-separated numbers.');
            newData.dpTable = nums;
            store.setExplanationText(`Custom DP table set with ${nums.length} values. Use the Start button to begin.`);
        } else {
            throw new Error(`Custom data not supported for ${type} type or algorithm ${currentAlgorithm}.`);
        }
        store.setData(newData);
        store.toggleModal();
    } catch (e) {
        alert(`Error parsing data: ${e.message}`);
    }
};

    const algoProps = getAlgorithmProps();

    return (
  <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6 font-sans text-gray-200">
    <div className="bg-gray-800 bg-opacity-90 backdrop-blur-md p-8 rounded-2xl shadow-2xl max-w-7xl mx-auto border border-gray-700">
      <h1 className="text-5xl font-extrabold text-center text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 mb-8 drop-shadow-lg">
        {algoProps.name} Visualization ✨
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 p-6 bg-gray-900 bg-opacity-70 rounded-xl shadow-inner border border-gray-700">
        <div>
          <label className="block text-sm font-semibold text-gray-300 mb-2">Algorithm Pattern</label>
          <select
            value={currentAlgorithm}
            onChange={(e) => store.setCurrentAlgorithm(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-200 focus:ring-2 focus:ring-cyan-500 focus:outline-none"
            disabled={isVisualizing}
          >
            {Object.entries(algorithms).map(([key, value]) => (
              <option key={key} value={key}>
                {value.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-300 mb-2">Animation Speed</label>
          <input
            type="range"
            min="0"
            max="11000"
            step="1000"
            value={animationSpeed}
            onChange={(e) => store.setAnimationSpeed(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg cursor-pointer accent-cyan-500"
            disabled={!isAutomated || isVisualizing}
          />
        </div>

        <div className="flex flex-col items-center space-y-4">
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-gray-400">Manual</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={isAutomated}
                onChange={(e) => store.setIsAutomated(e.target.checked)}
                className="sr-only peer"
                disabled={isVisualizing}
              />
              <div className="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:bg-cyan-500 peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all"></div>
            </label>
            <span className="text-sm font-medium text-gray-400">Automate</span>
          </div>

          <div className="flex items-center space-x-4 w-full">
            {!isAutomated ? (
              <>
                <button
                  onClick={handlePrev}
                  className="flex-1 bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-gray-500 transition disabled:bg-gray-500"
                  disabled={currentStepIndex.current <= 0}
                >
                  Previous
                </button>
                <button
                  onClick={!isVisualizing ? startVisualization : handleNext}
                  className="flex-1 bg-cyan-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-cyan-500 transition disabled:bg-cyan-400"
                  disabled={isComplete}
                >
                  {!isVisualizing ? 'Start' : 'Next'}
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={!isVisualizing ? startVisualization : () => store.setIsPaused(!isPaused)}
                  className="flex-1 bg-cyan-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-cyan-500 transition disabled:bg-cyan-400"
                  disabled={isComplete}
                >
                  {isVisualizing && !isPaused ? 'Pause' : isPaused ? 'Resume' : 'Start'}
                </button>
                {isVisualizing && (
                  <button
                    onClick={stopVisualization}
                    className="flex-1 bg-red-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-red-500 transition"
                  >
                    Stop
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex justify-center mb-8">
        <button
          onClick={toggleModal}
          className="bg-purple-600 text-white font-semibold py-2 px-6 rounded-lg hover:bg-purple-500 transition shadow-lg"
          disabled={isVisualizing}
        >
          Enter Custom Data
        </button>
      </div>

      {showModal && (
        <CustomDataModal
          isOpen={showModal}
          onSubmit={handleCustomDataSubmit}
          onClose={toggleModal}
          algoType={algoProps.type}
          currentAlgorithm={currentAlgorithm}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 bg-gray-900 border border-gray-700 rounded-xl shadow-lg min-h-[400px] p-4 flex overflow-auto">
          {algoProps.type === 'array' && <ArrayVisualizer array={array} highlights={highlights} />}
          {algoProps.type === 'board' && <BoardVisualizer board={board} highlights={highlights} />}
          {algoProps.type === 'graph' && <GraphVisualizer graph={graph} highlights={highlights} />}
          {algoProps.type === 'dp' && <DPVisualizer table={dpTable} highlights={highlights} />}
          {algoProps.type === 'linked-list' && <LinkedListVisualizer head={linkedList} highlights={highlights} />}
          {algoProps.type === 'tree' &&
            (currentAlgorithm === 'heap' ? (
              <HeapVisualizer heap={heap} highlights={highlights} />
            ) : currentAlgorithm === 'trie' ? (
              <TrieVisualizer trie={trie} highlights={highlights} />
            ) : (
              <BSTVisualizer trie={trie} highlights={highlights} />
            ))}
          {algoProps.type === 'other' && <BitVisualizer number={bitNumber} explanation={explanationText} highlights={highlights} />}
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-gray-800 p-6 rounded-xl shadow-md border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-3">Step-by-Step Explanation</h3>
            <p className="text-gray-300 min-h-[4rem]">{explanationText}</p>
            <div className="mt-4">
              <span className="text-sm text-gray-400">Step {stepCount}</span>
              <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
                <div
                  className="bg-cyan-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${isComplete ? 100 : (stepCount / (stepCount + 1)) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-xl shadow-md border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-3">Complexity</h3>
            <div className="grid grid-cols-2 gap-x-4 text-gray-300">
              <div>
                <span className="font-semibold">Time:</span> {algoProps.complexity.time}
              </div>
              <div>
                <span className="font-semibold">Space:</span> {algoProps.complexity.space}
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-xl shadow-md border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-3">Practice on LeetCode</h3>
            <ul className="space-y-2 list-disc list-inside text-blue-400">
              {algoProps.problems.map((p) => (
                <li key={p.name}>
                  <a href={p.url} target="_blank" rel="noopener noreferrer" className="hover:underline hover:text-blue-300">
                    {p.name} →
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="mt-10">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold text-white">Code Implementation</h2>
          <div className="flex space-x-2 bg-gray-700 p-1 rounded-lg">
            {['python', 'java', 'cpp'].map((lang) => (
              <button
                key={lang}
                onClick={() => store.setCurrentLanguage(lang)}
                className={`px-4 py-2 text-sm font-semibold rounded-md transition ${
                  store.currentLanguage === lang
                    ? 'bg-cyan-600 text-white shadow'
                    : 'bg-gray-600 text-gray-200 hover:bg-gray-500'
                }`}
              >
                {lang === 'cpp' ? 'C++' : lang.charAt(0).toUpperCase() + lang.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <pre className="bg-gray-900 text-green-300 p-6 rounded-xl overflow-x-auto text-sm shadow-inner border border-gray-700">
          <code>{algoProps.code[store.currentLanguage] || algoProps.pseudocode}</code>
        </pre>
      </div>
    </div>
    <footer className="mt-16 text-center text-sm text-gray-500">
  <div className="border-t border-gray-700 pt-6">
    © 2025 <span className="font-semibold text-white">AlgoVis</span>. Developed by{' '}
    <a
      href="https://tutumtetwa.com"
      target="_blank"
      rel="noopener noreferrer"
      className="text-cyan-400 hover:underline"
    >
      Tutu
    </a>
    .<br />
    All rights reserved.
    </div>
    </footer>

  </div>
  
);

}