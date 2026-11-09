# AlgoVis: Interactive Algorithm Visualizer

**Live:** https://algovis-iota.vercel.app

Step through common interview patterns one operation at a time. Each problem animates its data structure and highlights the line of code that is running, so you can see *why* the algorithm works rather than memorize it.

## What's inside
- **15 patterns, 60+ problems:** two pointers, sliding window, Kadane's, heaps / priority queues, tries, BFS, BST traversal, backtracking, dynamic programming, bit manipulation, linked lists, graphs and grid/board search
- **Custom visualizers per structure:** arrays, linked lists, heaps, tries, graphs, DP tables, bit strings and 2D boards (`src/*Visualizer.js`)
- **Your own input:** run any visualization on custom data
- **Playback controls:** step forward and back, play or pause, plus pan and zoom on large structures

## Stack
React 19 · D3 · Zustand · Tailwind · `react-zoom-pan-pinch` · deployed on Vercel

## Run locally
```bash
npm install
npm start          # http://localhost:3000
```
