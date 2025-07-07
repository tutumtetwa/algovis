// src/algorithm-data.js

export const algorithms = {
    // --- Foundational Array/String Patterns ---
    twoPointers: {
        name: "Two Pointers",
        type: "array",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(1)" },
        problems: [
            { name: "Two Sum II - Input Array Is Sorted", url: "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/" },
            { name: "Container With Most Water", url: "https://leetcode.com/problems/container-with-most-water/" },
            { name: "3Sum", url: "https://leetcode.com/problems/3sum/" },
            { name: "Valid Palindrome", url: "https://leetcode.com/problems/valid-palindrome/" },
            { name: "Trapping Rain Water", url: "https://leetcode.com/problems/trapping-rain-water/" },
            { name: "Remove Duplicates from Sorted Array", url: "https://leetcode.com/problems/remove-duplicates-from-sorted-array/" },
            { name: "Sort Colors", url: "https://leetcode.com/problems/sort-colors/" }
        ],
        pseudocode: `function twoPointers(arr):
  left ← 0
  right ← length(arr) - 1
  while left < right:
    // do some comparison or logic
    if condition is met:
      // move one or both pointers
      left ← left + 1
    else:
      right ← right - 1`,
        code: {
            python: `def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []`,
            java: `public int[] twoSum(int[] numbers, int target) {
    int left = 0, right = numbers.length - 1;
    while (left < right) {
        int sum = numbers[left] + numbers[right];
        if (sum == target) {
            return new int[]{left + 1, right + 1};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return new int[]{-1, -1};
}`,
            cpp: `vector<int> twoSum(vector<int>& numbers, int target) {
    int left = 0, right = numbers.size() - 1;
    while (left < right) {
        int sum = numbers[left] + numbers[right];
        if (sum == target) {
            return {left + 1, right + 1};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return {};
}`
        }
    },
    slidingWindow: {
        name: "Sliding Window",
        type: "array",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(k) or O(1)" },
        problems: [
            { name: "Longest Substring Without Repeating Characters", url: "https://leetcode.com/problems/longest-substring-without-repeating-characters/" },
            { name: "Minimum Size Subarray Sum", url: "https://leetcode.com/problems/minimum-size-subarray-sum/" },
            { name: "Best Time to Buy and Sell Stock", url: "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/" },
            { name: "Sliding Window Maximum", url: "https://leetcode.com/problems/sliding-window-maximum/" },
            { name: "Permutation in String", url: "https://leetcode.com/problems/permutation-in-string/" },
            { name: "Longest Repeating Character Replacement", url: "https://leetcode.com/problems/longest-repeating-character-replacement/" },
            { name: "Minimum Window Substring", url: "https://leetcode.com/problems/minimum-window-substring/" }
        ],
        pseudocode: `function slidingWindow(arr, k):
  left ← 0
  current_sum ← 0
  max_sum ← -INFINITY
  for right from 0 to length(arr) - 1:
    current_sum ← current_sum + arr[right]
    if (right - left + 1) is equal to k:
      max_sum ← max(max_sum, current_sum)
      current_sum ← current_sum - arr[left]
      left ← left + 1
  return max_sum`,
        code: {
            python: `def find_max_average(nums, k):
    max_sum = float('-inf')
    current_sum = 0
    window_start = 0
    for window_end in range(len(nums)):
        current_sum += nums[window_end]
        if window_end >= k - 1:
            max_sum = max(max_sum, current_sum)
            current_sum -= nums[window_start]
            window_start += 1
    return max_sum / k`,
            java: `public double findMaxAverage(int[] nums, int k) {
    double sum = 0;
    for (int i = 0; i < k; i++) {
        sum += nums[i];
    }
    double maxSum = sum;
    for (int i = k; i < nums.length; i++) {
        sum += nums[i] - nums[i - k];
        maxSum = Math.max(maxSum, sum);
    }
    return maxSum / k;
}`,
            cpp: `double findMaxAverage(vector<int>& nums, int k) {
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        sum += nums[i];
    }
    double maxSum = sum;
    for (int i = k; i < nums.size(); ++i) {
        sum += nums[i] - nums[i - k];
        maxSum = max(maxSum, sum);
    }
    return maxSum / k;
}`
        }
    },
    kadanes: {
        name: "Kadane's Algorithm",
        type: "array",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(1)" },
        problems: [
            { name: "Maximum Subarray", url: "https://leetcode.com/problems/maximum-subarray/" },
            { name: "Maximum Product Subarray", url: "https://leetcode.com/problems/maximum-product-subarray/" },
            { name: "Maximum Sum Circular Subarray", url: "https://leetcode.com/problems/maximum-sum-circular-subarray/" },
            { name: "Maximum Subarray Sum with One Deletion", url: "https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/" },
            { name: "Maximum Absolute Sum of Any Subarray", url: "https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/" },
            { name: "K-Concatenation Maximum Sum", url: "https://leetcode.com/problems/k-concatenation-maximum-sum/" },
            { name: "Largest Sum of Averages", url: "https://leetcode.com/problems/largest-sum-of-averages/" }
        ],
        pseudocode: `function kadane(arr):
  max_so_far ← -INFINITY
  max_ending_here ← 0
  for each element x in arr:
    max_ending_here ← max_ending_here + x
    if max_so_far < max_ending_here:
      max_so_far ← max_ending_here
    if max_ending_here < 0:
      max_ending_here ← 0
  return max_so_far`,
        code: {
            python: `def max_subarray(nums):
    max_so_far = nums[0]
    max_ending_here = nums[0]
    for i in range(1, len(nums)):
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far`,
            java: `public int maxSubArray(int[] nums) {
    int maxSoFar = nums[0];
    int maxEndingHere = nums[0];
    for (int i = 1; i < nums.length; i++) {
        maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
        maxSoFar = Math.max(maxSoFar, maxEndingHere);
    }
    return maxSoFar;
}`,
            cpp: `int maxSubArray(vector<int>& nums) {
    int maxSoFar = nums[0];
    int maxEndingHere = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        maxEndingHere = max(nums[i], maxEndingHere + nums[i]);
        maxSoFar = max(maxSoFar, maxEndingHere);
    }
    return maxSoFar;
}`
        }
    },

    // --- Core Data Structures ---
    heap: {
        name: "Heap (Priority Queue)",
        type: "tree",
        isImplemented: true,
        complexity: { time: "O(log n) insert/extract", space: "O(n)" },
        problems: [
            { name: "Kth Largest Element in an Array", url: "https://leetcode.com/problems/kth-largest-element-in-an-array/" },
            { name: "Find Median from Data Stream", url: "https://leetcode.com/problems/find-median-from-data-stream/" },
            { name: "Top K Frequent Elements", url: "https://leetcode.com/problems/top-k-frequent-elements/" },
            { name: "Merge K Sorted Lists", url: "https://leetcode.com/problems/merge-k-sorted-lists/" },
            { name: "Kth Smallest Element in a Sorted Matrix", url: "https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/" },
            { name: "Task Scheduler", url: "https://leetcode.com/problems/task-scheduler/" },
            { name: "Find K Pairs with Smallest Sums", url: "https://leetcode.com/problems/find-k-pairs-with-smallest-sums/" }
        ],
        pseudocode: `// Build a Min-Heap from an array
procedure buildHeap(array):
  n = length(array)
  // Start from the last non-leaf node and heapify down
  for i from floor(n / 2) - 1 down to 0:
    heapifyDown(array, n, i)

procedure heapifyDown(array, n, i):
  smallest = i
  left = 2*i + 1
  right = 2*i + 2

  if left < n and array[left] < array[smallest]:
    smallest = left
  
  if right < n and array[right] < array[smallest]:
    smallest = right
    
  if smallest is not i:
    swap(array[i], array[smallest])
    heapifyDown(array, n, smallest)`,
        code: {
            python: `import heapq
# Python's heapq module provides a min-heap implementation.
# To build a heap from a list:
heapq.heapify(list)

# To push an item:
heapq.heappush(heap, item)

# To pop the smallest item:
heapq.heappop(heap)`,
            java: `// Java's PriorityQueue is a min-heap by default.
PriorityQueue<Integer> minHeap = new PriorityQueue<>();

// To add an item:
minHeap.add(10);

// To get the smallest item without removing:
minHeap.peek();

// To remove and get the smallest item:
minHeap.poll();`,
            cpp: `// C++'s priority_queue is a max-heap by default.
// To create a min-heap:
std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;

// To add an item:
minHeap.push(10);

// To get the smallest item:
minHeap.top();

// To remove the smallest item:
minHeap.pop();`
        }
    },
    trie: {
        name: "Trie (Prefix Tree)",
        type: "tree",
        isImplemented: true,
        complexity: { time: "O(L) insert/search", space: "O(N*L)" },
        problems: [
            { name: "Implement Trie (Prefix Tree)", url: "https://leetcode.com/problems/implement-trie-prefix-tree/" },
            { name: "Design Add and Search Words Data Structure", url: "https://leetcode.com/problems/design-add-and-search-words-data-structure/" },
            { name: "Word Search II", url: "https://leetcode.com/problems/word-search-ii/" },
            { name: "Maximum XOR of Two Numbers in an Array", url: "https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/" },
            { name: "Replace Words", url: "https://leetcode.com/problems/replace-words/" },
            { name: "Word Squares", url: "https://leetcode.com/problems/word-squares/" },
            { name: "Palindrome Pairs", url: "https://leetcode.com/problems/palindrome-pairs/" }
        ],
        pseudocode: `TrieNode:
  children: map of character to TrieNode
  isEndOfWord: boolean

procedure insert(word):
  node ← root
  for each character ch in word:
    if ch is not in node.children:
      node.children[ch] ← new TrieNode()
    node ← node.children[ch]
  node.isEndOfWord ← true

procedure search(word):
  node ← root
  for each character ch in word:
    if ch is not in node.children:
      return false
    node ← node.children[ch]
  return node.isEndOfWord`,
        code: {
            python: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word`,
            java: `class TrieNode {
    public Map<Character, TrieNode> children = new HashMap<>();
    public boolean isEndOfWord = false;
}

class Trie {
    private TrieNode root;
    public Trie() { root = new TrieNode(); }
    
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            node.children.putIfAbsent(c, new TrieNode());
            node = node.children.get(c);
        }
        node.isEndOfWord = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (!node.children.containsKey(c)) {
                return false;
            }
            node = node.children.get(c);
        }
        return node.isEndOfWord;
    }
}`,
            cpp: `struct TrieNode {
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord = false;
};

class Trie {
public:
    TrieNode* root;
    Trie() { root = new TrieNode(); }
    
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->isEndOfWord = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return node->isEndOfWord;
    }
};`
        }
    },

    // --- Core Traversal Patterns ---
    bfs: {
        name: "Breadth-First Search (BFS)",
        type: "graph",
        isImplemented: true,
        complexity: { time: "O(V+E)", space: "O(V)" },
        problems: [
            { name: "Binary Tree Level Order Traversal", url: "https://leetcode.com/problems/binary-tree-level-order-traversal/" },
            { name: "Minimum Depth of Binary Tree", url: "https://leetcode.com/problems/minimum-depth-of-binary-tree/" },
            { name: "Number of Islands", url: "https://leetcode.com/problems/number-of-islands/" },
            { name: "Rotting Oranges", url: "https://leetcode.com/problems/rotting-oranges/" },
            { name: "Word Ladder", url: "https://leetcode.com/problems/word-ladder/" },
            { name: "01 Matrix", url: "https://leetcode.com/problems/01-matrix/" },
            { name: "Shortest Path in Binary Matrix", url: "https://leetcode.com/problems/shortest-path-in-binary-matrix/" }
        ],
        pseudocode: `procedure BFS(graph, start_node):
  let Q be a queue
  let visited be a set
  
  Q.enqueue(start_node)
  visited.add(start_node)
  
  while Q is not empty:
    vertex = Q.dequeue()
    process(vertex)
    for each neighbor of vertex:
      if neighbor has not been visited:
        visited.add(neighbor)
        Q.enqueue(neighbor)`,
        code: {
            python: `from collections import deque
def levelOrder(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
    return result`,
            java: `public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();
        for (int i = 0; i < levelSize; i++) {
            TreeNode node = queue.poll();
            currentLevel.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        result.add(currentLevel);
    }
    return result;
}`,
            cpp: `vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        for (int i = 0; i < levelSize; ++i) {
            TreeNode* node = q.front();
            q.pop();
            currentLevel.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(currentLevel);
    }
    return result;
}`
        }
    },
    bstTraversal: {
        name: "BST Traversal",
        type: "tree",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(h)" },
        problems: [
            { name: "Binary Tree Inorder Traversal", url: "https://leetcode.com/problems/binary-tree-inorder-traversal/" },
            { name: "Validate Binary Search Tree", url: "https://leetcode.com/problems/validate-binary-search-tree/" },
            { name: "Binary Tree Preorder Traversal", url: "https://leetcode.com/problems/binary-tree-preorder-traversal/" },
            { name: "Binary Tree Postorder Traversal", url: "https://leetcode.com/problems/binary-tree-postorder-traversal/" },
            { name: "Kth Smallest Element in a BST", url: "https://leetcode.com/problems/kth-smallest-element-in-a-bst/" },
            { name: "Convert Sorted Array to Binary Search Tree", url: "https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/" },
            { name: "Lowest Common Ancestor of a Binary Search Tree", url: "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/" }
        ],
        pseudocode: `procedure inOrder(node):
  if node is null, return
  inOrder(node.left)
  process(node)
  inOrder(node.right)

procedure preOrder(node):
  if node is null, return
  process(node)
  preOrder(node.left)
  preOrder(node.right)

procedure postOrder(node):
  if node is null, return
  postOrder(node.left)
  postOrder(node.right)
  process(node)`,
        code: {
            python: `def inorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res`,
            java: `public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    helper(root, res);
    return res;
}
private void helper(TreeNode node, List<Integer> res) {
    if (node == null) return;
    helper(node.left, res);
    res.add(node.val);
    helper(node.right, res);
}`,
            cpp: `vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res;
    function<void(TreeNode*)> dfs = 
        [&](TreeNode* node) {
        if (!node) return;
        dfs(node->left);
        res.push_back(node->val);
        dfs(node->right);
    };
    dfs(root);
    return res;
}`
        }
    },

    // --- Recursive Patterns ---
    backtracking: {
        name: "Backtracking",
        type: "board",
        isImplemented: true,
        complexity: { time: "O(N!) or O(2^N)", space: "O(N)" },
        problems: [
            { name: "Subsets", url: "https://leetcode.com/problems/subsets/" },
            { name: "Combination Sum", url: "https://leetcode.com/problems/combination-sum/" },
            { name: "Permutations", url: "https://leetcode.com/problems/permutations/" },
            { name: "N-Queens", url: "https://leetcode.com/problems/n-queens/" },
            { name: "Word Search", url: "https://leetcode.com/problems/word-search/" },
            { name: "Generate Parentheses", url: "https://leetcode.com/problems/generate-parentheses/" },
            { name: "Letter Combinations of a Phone Number", url: "https://leetcode.com/problems/letter-combinations-of-a-phone-number/" }
        ],
        pseudocode: `procedure backtrack(state, choices):
  if state is a solution:
    add state to solutions
    return
  
  for each choice in choices:
    if choice is valid:
      make choice // modify state
      backtrack(new_state, new_choices)
      undo choice // revert state`,
        code: {
            python: `def subsets(nums):
    res = []
    subset = []
    def backtrack(i):
        if i >= len(nums):
            res.append(subset.copy())
            return
        # Decision to include nums[i]
        subset.append(nums[i])
        backtrack(i + 1)
        # Decision NOT to include nums[i]
        subset.pop()
        backtrack(i + 1)
    backtrack(0)
    return res`,
            java: `public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(result, new ArrayList<>(), nums, 0);
    return result;
}
private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, int start){
    list.add(new ArrayList<>(tempList));
    for(int i = start; i < nums.length; i++){
        tempList.add(nums[i]);
        backtrack(list, tempList, nums, i + 1);
        tempList.remove(tempList.size() - 1);
    }
}`,
            cpp: `vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> subset;
    function<void(int)> backtrack = 
        [&](int i) {
        if (i == nums.size()) {
            result.push_back(subset);
            return;
        }
        // Exclude nums[i]
        backtrack(i + 1);
        // Include nums[i]
        subset.push_back(nums[i]);
        backtrack(i + 1);
        subset.pop_back();
    };
    backtrack(0);
    return result;
}`
        }
    },
    
    // --- Dynamic Programming ---
    dpFib: {
        name: "DP (1D)",
        type: "dp",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(n) or O(1)" },
        problems: [
            { name: "Climbing Stairs", url: "https://leetcode.com/problems/climbing-stairs/" },
            { name: "House Robber", url: "https://leetcode.com/problems/house-robber/" },
            { name: "Coin Change", url: "https://leetcode.com/problems/coin-change/" },
            { name: "Longest Increasing Subsequence", url: "https://leetcode.com/problems/longest-increasing-subsequence/" },
            { name: "Word Break", url: "https://leetcode.com/problems/word-break/" },
            { name: "Partition Equal Subset Sum", url: "https://leetcode.com/problems/partition-equal-subset-sum/" },
            { name: "Decode Ways", url: "https://leetcode.com/problems/decode-ways/" }
        ],
        pseudocode: `function climbStairs(n):
  if n <= 2 return n
  dp = array of size n + 1
  dp[1] = 1
  dp[2] = 2
  for i from 3 to n:
    dp[i] = dp[i-1] + dp[i-2]
  return dp[n]`,
        code: {
            python: `def climbStairs(n):
    if n <= 2:
        return n
    prev1, prev2 = 2, 1
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    return prev1`,
            java: `public int climbStairs(int n) {
    if (n <= 2) return n;
    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}`,
            cpp: `int climbStairs(int n) {
    if (n <= 2) return n;
    vector<int> dp(n + 1);
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}`
        }
    },

    // --- Advanced Graph ---
    dijkstra: {
        name: "Dijkstra's Algorithm",
        type: "graph",
        isImplemented: true,
        complexity: { time: "O(E log V)", space: "O(V)" },
        problems: [
            { name: "Network Delay Time", url: "https://leetcode.com/problems/network-delay-time/" },
            { name: "Path with Minimum Effort", url: "https://leetcode.com/problems/path-with-minimum-effort/" },
            { name: "The Maze II", url: "https://leetcode.com/problems/the-maze-ii/" },
            { name: "Cheapest Flights Within K Stops", url: "https://leetcode.com/problems/cheapest-flights-within-k-stops/" },
            { name: "Swim in Rising Water", url: "https://leetcode.com/problems/swim-in-rising-water/" },
            { name: "Path with Maximum Probability", url: "https://leetcode.com/problems/path-with-maximum-probability/" },
            { name: "Number of Ways to Arrive at Destination", url: "https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/" }
        ],
        pseudocode: `function Dijkstra(Graph, source):
  dist[source] ← 0
  create vertex priority queue Q
  for each vertex v in Graph:
    if v ≠ source:
      dist[v] ← INFINITY
    Q.add_with_priority(v, dist[v])
  
  while Q is not empty:
    u ← Q.extract_min()
    for each neighbor v of u:
      alt ← dist[u] + length(u, v)
      if alt < dist[v]:
        dist[v] ← alt
        Q.decrease_priority(v, alt)
  return dist[]`,
        code: {
            python: `import heapq
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node].items():
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances`,
            java: `// See previous complete example`,
            cpp: `// See previous complete example`
        }
    },
    topologicalSort: {
        name: "Topological Sort",
        type: "graph",
        isImplemented: true,
        complexity: { time: "O(V+E)", space: "O(V)" },
        problems: [
            { name: "Course Schedule", url: "https://leetcode.com/problems/course-schedule/" },
            { name: "Course Schedule II", url: "https://leetcode.com/problems/course-schedule-ii/" },
            { name: "Alien Dictionary", url: "https://leetcode.com/problems/alien-dictionary/" },
            { name: "Minimum Height Trees", url: "https://leetcode.com/problems/minimum-height-trees/" },
            { name: "Sequence Reconstruction", url: "https://leetcode.com/problems/sequence-reconstruction/" },
            { name: "Build a Matrix With Conditions", url: "https://leetcode.com/problems/build-a-matrix-with-conditions/" },
            { name: "Sort Items by Groups Respecting Dependencies", url: "https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/" }
        ],
        pseudocode: `// Kahn's Algorithm
procedure topologicalSort(graph):
  in_degree ← map of node to integer, initialized to 0
  for each node u in graph:
    for each neighbor v of u:
      in_degree[v] ← in_degree[v] + 1
  
  queue ← queue of all nodes with in_degree 0
  result ← empty list
  
  while queue is not empty:
    u ← queue.dequeue()
    result.append(u)
    for each neighbor v of u:
      in_degree[v] ← in_degree[v] - 1
      if in_degree[v] is 0:
        queue.enqueue(v)
        
  if length(result) is equal to number of nodes:
    return result
  else:
    return "Graph has a cycle"`,
        code: {
            python: `from collections import deque
def findOrder(numCourses, prerequisites):
    adj = {i: [] for i in range(numCourses)}
    in_degree = [0] * numCourses
    for course, prereq in prerequisites:
        adj[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []
    
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    return result if len(result) == numCourses else []`,
            java: `public int[] findOrder(int numCourses, int[][] prerequisites) {
    Map<Integer, List<Integer>> adj = new HashMap<>();
    int[] inDegree = new int[numCourses];
    for (int[] p : prerequisites) {
        adj.computeIfAbsent(p[1], k -> new ArrayList<>()).add(p[0]);
        inDegree[p[0]]++;
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) queue.offer(i);
    }
    int[] result = new int[numCourses];
    int i = 0;
    while (!queue.isEmpty()) {
        int u = queue.poll();
        result[i++] = u;
        for (int v : adj.getOrDefault(u, new ArrayList<>())) {
            inDegree[v]--;
            if (inDegree[v] == 0) queue.offer(v);
        }
    }
    return i == numCourses ? result : new int[0];
}`,
            cpp: `vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> adj(numCourses);
    vector<int> inDegree(numCourses, 0);
    for (const auto& p : prerequisites) {
        adj[p[1]].push_back(p[0]);
        inDegree[p[0]]++;
    }
    queue<int> q;
    for (int i = 0; i < numCourses; ++i) {
        if (inDegree[i] == 0) q.push(i);
    }
    vector<int> result;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        result.push_back(u);
        for (int v : adj[u]) {
            if (--inDegree[v] == 0) q.push(v);
        }
    }
    return result.size() == numCourses ? result : vector<int>();
}`
        }
    },
    unionFind: {
        name: "Union-Find (DSU)",
        type: "graph",
        isImplemented: true,
        complexity: { time: "α(n) (nearly constant)", space: "O(n)" },
        problems: [
            { name: "Number of Connected Components in an Undirected Graph", url: "https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/" },
            { name: "Graph Valid Tree", url: "https://leetcode.com/problems/graph-valid-tree/" },
            { name: "Redundant Connection", url: "https://leetcode.com/problems/redundant-connection/" },
            { name: "Accounts Merge", url: "https://leetcode.com/problems/accounts-merge/" },
            { name: "Number of Provinces", url: "https://leetcode.com/problems/number-of-provinces/" },
            { name: "Satisfiability of Equality Equations", url: "https://leetcode.com/problems/satisfiability-of-equality-equations/" },
            { name: "The Earliest Moment When Everyone Becomes Friends", url: "https://leetcode.com/problems/the-earliest-moment-when-everyone-becomes-friends/" }
        ],
        pseudocode: `class UnionFind:
  parent: array of integers
  
  constructor(size):
    parent ← new array of size, where parent[i] = i
    
  function find(i):
    if parent[i] is i:
      return i
    parent[i] ← find(parent[i]) // Path compression
    return parent[i]
    
  function union(i, j):
    root_i ← find(i)
    root_j ← find(j)
    if root_i is not root_j:
      parent[root_j] ← root_i
      return true // They were not connected
    return false // They were already connected`,
        code: {
            python: `class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
    
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i]) # Path compression
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False`,
            java: `class UnionFind {
    private int[] parent;
    private int[] rank;

    public UnionFind(int size) {
        parent = new int[size];
        rank = new int[size];
        for (int i = 0; i < size; i++) {
            parent[i] = i;
            rank[i] = 1;
        }
    }

    public int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]); // Path compression
    }

    public boolean union(int i, int j) {
        int rootI = find(i);
        int rootJ = find(j);
        if (rootI != rootJ) {
            // Union by rank
            if (rank[rootI] > rank[rootJ]) {
                parent[rootJ] = rootI;
            } else if (rank[rootI] < rank[rootJ]) {
                parent[rootI] = rootJ;
            } else {
                parent[rootJ] = rootI;
                rank[rootI]++;
            }
            return true;
        }
        return false;
    }
}`,
            cpp: `class UnionFind {
public:
    vector<int> parent;
    vector<int> rank;
    UnionFind(int size) {
        parent.resize(size);
        iota(parent.begin(), parent.end(), 0);
        rank.assign(size, 1);
    }

    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]); // Path compression
    }

    bool unite(int i, int j) {
        int rootI = find(i);
        int rootJ = find(j);
        if (rootI != rootJ) {
            // Union by rank
            if (rank[rootI] > rank[rootJ]) {
                parent[rootJ] = rootI;
            } else if (rank[rootI] < rank[rootJ]) {
                parent[rootI] = rootJ;
            } else {
                parent[rootJ] = rootI;
                rank[rootI]++;
            }
            return true;
        }
        return false;
    }
};`
        }
    },

    // --- Linked List Patterns ---
    linkedListReversal: {
        name: "Linked List Reversal",
        type: "linked-list",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(1)" },
        problems: [
            { name: "Reverse Linked List", url: "https://leetcode.com/problems/reverse-linked-list/" },
            { name: "Reverse Linked List II", url: "https://leetcode.com/problems/reverse-linked-list-ii/" },
            { name: "Palindrome Linked List", url: "https://leetcode.com/problems/palindrome-linked-list/" },
            { name: "Swap Nodes in Pairs", url: "https://leetcode.com/problems/swap-nodes-in-pairs/" },
            { name: "Reverse Nodes in k-Group", url: "https://leetcode.com/problems/reverse-nodes-in-k-group/" },
            { name: "Add Two Numbers II", url: "https://leetcode.com/problems/add-two-numbers-ii/" },
            { name: "Reorder List", url: "https://leetcode.com/problems/reorder-list/" }
        ],
        pseudocode: `function reverseList(head):
  prev ← null
  curr ← head
  while curr is not null:
    next_temp ← curr.next
    curr.next ← prev
    prev ← curr
    curr ← next_temp
  return prev`,
        code: {
            python: `def reverseList(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev`,
            java: `public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}`,
            cpp: `ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    while (curr) {
        ListNode* nextTemp = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}`
        }
    },
    floydsCycle: {
        name: "Floyd's Cycle Detection",
        type: "linked-list",
        isImplemented: true,
        complexity: { time: "O(n)", space: "O(1)" },
        problems: [
            { name: "Linked List Cycle", url: "https://leetcode.com/problems/linked-list-cycle/" },
            { name: "Linked List Cycle II", url: "https://leetcode.com/problems/linked-list-cycle-ii/" },
            { name: "Find the Duplicate Number", url: "https://leetcode.com/problems/find-the-duplicate-number/" },
            { name: "Happy Number", url: "https://leetcode.com/problems/happy-number/" },
            { name: "Middle of the Linked List", url: "https://leetcode.com/problems/middle-of-the-linked-list/" },
            { name: "Circular Array Loop", url: "https://leetcode.com/problems/circular-array-loop/" },
            { name: "Delete the Middle Node of a Linked List", url: "https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/" }
        ],
        pseudocode: `// Tortoise and Hare Algorithm
function hasCycle(head):
  if head is null:
    return false
  slow ← head
  fast ← head.next
  while slow is not fast:
    if fast is null or fast.next is null:
      return false
    slow ← slow.next
    fast ← fast.next.next
  return true`,
        code: {
            python: `def hasCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`,
            java: `public boolean hasCycle(ListNode head) {
    if (head == null) return false;
    ListNode slow = head;
    ListNode fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) {
            return true;
        }
    }
    return false;
}`,
            cpp: `bool hasCycle(ListNode *head) {
    if (!head) return false;
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            return true;
        }
    }
    return false;
}`
        }
    },

    // --- Other ---
    bitManipulation: {
        name: "Bit Manipulation",
        type: "other",
        isImplemented: true,
        complexity: { time: "O(1) or O(log n)", space: "O(1)" },
        problems: [
            { name: "Single Number", url: "https://leetcode.com/problems/single-number/" },
            { name: "Number of 1 Bits", url: "https://leetcode.com/problems/number-of-1-bits/" },
            { name: "Counting Bits", url: "https://leetcode.com/problems/counting-bits/" },
            { name: "Reverse Bits", url: "https://leetcode.com/problems/reverse-bits/" },
            { name: "Missing Number", url: "https://leetcode.com/problems/missing-number/" },
            { name: "Sum of Two Integers", url: "https://leetcode.com/problems/sum-of-two-integers/" },
            { name: "Power of Two", url: "https://leetcode.com/problems/power-of-two/" }
        ],
        pseudocode: `// Example: Check if a number is a power of two
function isPowerOfTwo(n):
  if n <= 0:
    return false
  return (n AND (n - 1)) is 0

// Example: Count set bits (Hamming Weight)
function countSetBits(n):
  count ← 0
  while n > 0:
    n ← n AND (n - 1)
    count ← count + 1
  return count`,
        code: {
            python: `def hammingWeight(n: int) -> int:
    count = 0
    while n != 0:
        n &= (n - 1)
        count += 1
    return count`,
            java: `public int hammingWeight(int n) {
    int count = 0;
    while (n != 0) {
        n = n & (n - 1);
        count++;
    }
    return count;
}`,
            cpp: `int hammingWeight(uint32_t n) {
    int count = 0;
    while (n != 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}`
        }
    },
};