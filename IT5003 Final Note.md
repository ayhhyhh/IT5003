## Sort

### Basic

![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312062124281.png)

```python
from bisect import bisect_left, bisect_right
bisect_left(L, x) # return ip that L[0:ip] < x
bisect_right(L, x) # L[ip:] > x
```
### Bubble Sort

```
method bubbleSort(array A, integer N) // the standard version  
  for each R from N-1 down to 1 // repeat for N-1 iterations  
    for each index i in [0..R-1] // the 'unsorted region', O(N)  
      if A[i] > A[i+1] // these two are not in non-decreasing order  
        swap(a[i], a[i+1]) // swap them in O(1)
```

```python
def bubbleSort(A): # O(N^2) worst case (reverse sorted input), O(N) best case (sorted input)
    N = len(A)
    while N > 1: # at most n-1 passes
        swapped = False
        for i in range(N-1):
            if A[i] > A[i+1]:
                A[i], A[i+1] = A[i+1], A[i] # Python can swap variables like this
                swapped = True
        if not swapped: # optimization
            break
        N -= 1
    return A
```

### Selection Sort

```
method selectionSort(array A[], integer N)  
  for each L in [0..N-2] // O(**N**)  
    let X be the index of the minimum element in A[L..N-1] // O(**N**)  
    swap(A[X], A[L]) // O(1), X may be equal to L (no actual swap)
```

```python
def selectionSort(A): # O(N^2) for ALL cases...
    N = len(A)
    for L in range(N-1):
        smallest = L + A[L:].index(min(A[L:])) # BEWARE... this is O(N) not O(1)... we cannot find the smallest index of the minimum element of (N-L) items in O(1)
        A[smallest], A[L] = A[L], A[smallest] # Python can swap variables like this
    return A
```

### Insertion Sort

```
method insertionSort(array A[], integer N)  
  for i in [1..N-1] // O(N)  
    let X be A[i] // X is the next item to be inserted into A[0..i-1]  
    for j from i-1 down to 0 // this loop can be fast or slow  
      if A[j] > X  
        A[j+1] = A[j] // make a place for X  
      else  
        break  
    A[j+1] = X // insert X at index j+1
```

```python
def insertionSort(A): # O(N^2) worst case (reverse sorted input), O(N) best case (sorted input)
    N = len(A)
    for i in range(1, N): # O(N)
        X = A[i] # X is the item to be inserted
        j = i-1
        while j >= 0 and A[j] > X: # can be fast or slow
            A[j+1] = A[j] # make a place for X
            j -= 1
        A[j+1] = X # index j+1 is the insertion point
    return A
```

### Merge Sort

```
method mergeSort(array A, integer low, integer high)  
  // the array to be sorted is A[low..high]  
  if (low < high) // base case: low >= high (0 or 1 item)  
    int mid = (low+high) / 2	  
    mergeSort(a, low  , mid ) // divide into two halves  
    mergeSort(a, mid+1, high) // then recursively sort them  
    merge(a, low, mid, high) // conquer: the merge subroutine
```

```python
def mergeSort(A): # O(N log N) worst case for ALL cases :)
    N = len(A)
    if N == 1: # base case, it is trivial to sort a single element list
        return A # just do nothing, return the list as it is

    mid = N//2 # PS: The one in VisuAlgo has right sublist 1 bigger than the left sublist when N is odd
    left = A[:mid] # from start to before mid, if N is odd, left is one less than right
    right = A[mid:] # from mid to end
    left_sorted = mergeSort(left) # recursively sort the left sublist
    assert(left_sorted == left) # left is directly modified to its sorted version, so we do not need to assign the result into variable left
    mergeSort(right) # recursively sort the right sublist

    i, j, k = 0, 0, 0
    while i < len(left) and j < len(right): # both left and right not empty
        if left[i] <= right[j]:
            A[k] = left[i] # take from left
            i += 1
        else:
            A[k] = right[j] # take from right
            j += 1
        k += 1
    while i < len(left): # has leftover from left (right is empty)
        A[k] = left[i]
        k += 1
        i += 1
    while j < len(right): # has leftover from right (left is empty)
        A[k] = right[j]
        k += 1
        j += 1

    return A
```

### Quick Sort

```python
def quickSort(A, low, high): # expected O(N log N) worst case for ALL cases, the heavy time complexity analysis involving expected values are omitted
    if low < high:
        r = low + random.randrange(high-low+1) # a random index between [low..high]
        A[low], A[r] = A[r], A[low] # tada

        p = A[low] # p is the pivot
        m = low # S1 and S2 are initially empty
        for k in range(low+1, high+1): # expore the unknown region
            # case 2 (PATCHED solution to avoid TLE O(N^2) on input list with identical values)
            if A[k] < p or (A[k] == p and random.randrange(2) == 0):
                m += 1
                A[k], A[m] = A[m], A[k]
            # notice that we do nothing in case 1: A[k] > p
        A[low], A[m] = A[m], A[low] # final step, swap pivot with A[m]

        # a[low..high] ~> a[low..m-1], pivot, a[m+1..high]
        quickSort(A, low, m-1) # recursively sort left sublist
        # A[m] = pivot is already sorted after partition
        quickSort(A, m+1, high) # recursively sort right sublist

    return A
```

### Count Sort

```python
def count_sort(A):
	count = [0] * (max(A) + 1)
	for i in range(len(A)):
		count[A[i]] += 1
	L = []
	for i in range(count):
		while count[i]:
			count[i] -= 1
			L.append(i)
	return L
```

### Radix Sort

```python
def radix_sort(A):
	k = 10
	base = 1 << k
	for i in range(3):
		buckets = [[] for _ in range(base)]
		# put in the bucket
		for j in range(len(A)):
			buckets[(A[j] >> (i*k)) & (base-1)].append(A[j])
		# put out from bucket
		idx = 0
		for bucket in buckets:
			for e in bucket:
				A[idx] = e
				idx += 1
	return A
```

## Link List

### Basic

![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312062123929.png)

Sort on Link List:

- Option 1: Merge sort on Linked List: simple, during the merge process, we create a new linked list to merge two sorted shorter lists. Also O(N) for this merge of two sorted linked lists.
- Option 2: (Randomized) Quick sort on Linked List: not that hard either (assuming no duplicate first), we can pick a random element as pivot (probably in O(N), not in O(1)) and create a new linked list with just that pivot. We then grow to the left (add Head) if the next other element is smaller than the pivot or grow to the right (add Tail) otherwise. Also O(N) for this partition on linked list.

Pre/In/Post-Fix Expression:
- Pre/In/Post Travesal The Expression Tree
## Binary Heap

### Basic

- A binary tree with $N$ nodes has height $\log (N+1) - 1$
- 1-Based compact array can represent a binaryheap.
	- Parent():  i << 1
	- Left Child():   i >> 1
	- Right Child():  i >> 1 + 1
- Operations:
1. **Create(A)** - O(**N** log **N**) version (**N** calls of **Insert(v)** below)
2. **Create(A)** - O(**N**) version
3. **Insert(v)** in O(log **N**) — you are allowed to insert duplicates
4. 3 versions of **ExtractMax()**:
    1. Once, in O(log **N**)
    2. **K** times, i.e., **PartialSort()**, in O(**K** log **N**), or
    3. **N** times, i.e., **HeapSort()**, in O(**N** log **N)**
5. **UpdateKey(i, newv)** in O(log **N** if **i** is known)
6. **Delete(i)** in O(log **N** if **i** is known)

- Find findVerticesBiggerThanX(vertex, X):
```
findVerticesBiggerThanX(vertex, X): # pre-order traversal
	if (vertex.key > x) then
		output(vertex.key)
		findVerticesBiggerThanX(vertex.left, x)
		findVerticesBiggerThanX(vertex.right, x)
	end if
```

## Hash Table

```python
stdout_output, variables, st = [], {}, [set()]
n = int(input())
for _ in range(n):
    cmd = input().split(" ")
    if cmd[0] == "{":
        st.append(set())
    elif cmd[0] == "}":
        local_variables = st.pop()
        for v in local_variables:
            variables[v].pop()  
    elif cmd[0][0] == "D":
        _, variable_name, variable_type = cmd
        if variable_name not in st[-1]:
            st[-1].add(variable_name)
            if variables.get(variable_name):
                variables[variable_name].append(variable_type)
            else:
                variables[variable_name] = [variable_type]
        else:
            print("MULTIPLE DECLARATION")
            break
    elif cmd[0][0] == "T":
        _, variable_name = cmd
        if variables.get(variable_name):
            print(variables[variable_name][-1])
        else:
            print("UNDECLARED")
```
## Graph

So far, we can use DFS/BFS to solve a few graph traversal problem variants:

1. Reachability test,
2. Actually printing the traversal path,
3. Identifying/Counting/Labeling Connected Components (CCs) of undirected graphs,
4. Detecting if a graph is cyclic,
5. Topological Sort (only on DAGs),

### BFS

```python
def bfs(v):
    q = [v]
    while q:
        newq = []
        for i in q:
            visit(i)
            for neighbor in AL[i]:
                if not vis[q]:
                    vis[q] = 1
                    newq.append(neighbor)
        q = newq
```

### DFS

```python
def dfs(v):
    vis[v] = 1
    visit(v)
    for neighbor in AL[v]:
        if not vis[neighbor]:
            dfs(neighbor)
```

### Detect Cycle

```python
UNVISITED = -1
EXPLORED = -2
VISITED = -3

AL = []
dfs_num = [] # status unvisited, explored, visited
dfs_parent = [] # track parent 

def cycleCheck(u):
    global AL
    global dfs_num
    global dfs_parent
    dfs_num[u] = EXPLORED

    for v, w in AL[u]:
        if dfs_num[v] == UNVISITED:
            dfs_parent[v] = u
            cycleCheck(v)
        elif dfs_num[v] == EXPLORED:
            if v == dfs_parent[u]:
                print('Bidirectional Edge (%d, %d)-(%d, %d)' % (u, v, v, u))
            else:
                print('Back Edge (%d, %d) (Cycle)' % (u, v))
    dfs_num[u] = VISITED
for u in range(V):
    if dfs_num[u] == UNVISITED:
        cycleCheck(u)
```

### Topological Sort


```python
# DFS
UNVISITED = -1
VISITED = -2
AL = []
dfs_num = [UNVISITED] * V # status unvisited, visited
ts = [] # ts result  
# In fact, is post-order travesal
def toposort(u):
    global AL
    global dfs_num
    global ts
    dfs_num[u] = VISITED
    for v, w in AL[u]:
        if dfs_num[v] == UNVISITED:
            toposort(v)
    ts.append(u) 

for u in range(V):
    if dfs_num[u] == UNVISITED:
        toposort(u)
ts = ts[::-1]
```

```python
# BFS
from collections import deque
V = 10
AL = [[]]
indegree = [0]*V
for i in range(V): # 构造indegree数组
    for j, w in AL: indegree[j] += 1
q = deque([])
for v in range(V): # q维护indegree=0的点
    if indegree[v] == 0: q.append(v)
topo = []
while q:
    v = q.popleft()
    topo.append(v)
    for u, w in AL[v]:
        indegree[u] -= 1
        if indegree[u] == 0: q.append(u)
for v in range(V):
    if indegree[v] != 0: print("Exist Cycle")
```

Check if sequence is a topo sort.

```python
# BFS
from collections import deque
V = 10
AL = [[]]
indegree = [0]*V
for i in range(V): # 构造indegree数组
    for j, w in AL: indegree[j] += 1
topoToCheck = deque([])
isTopo = True
while topoToCheck:
    v = topoToCheck.popleft()
    if indegree[v] != 0:
        isTopo = False
        break
    for u, w in AL[v]:
        indegree[u] -= 1
if isTopo:
    print("Is topo")
else:
    print("Not a topo")
```
### SSSP

#### Relax

```
relax(u, v, w_u_v)  
  if D[v] > D[u]+w_u_v // if the path can be shortened  
    D[v] = D[u]+w_u_v // we 'relax' this edge  
    p[v] = u // remember/update the predecessor
```

#### Bellman-Ford

```
for i = 1 to |V|-1 // O(V) here, so O(V×E×1) = O(V×E)  
  for each edge(u, v) ∈ E // O(E) here, e.g. by using an Edge List  
    relax(u, v, w(u, v)) // O(1) here
```

```python
# Bellman Ford's routine, basically = relax all E edges V-1 times
dist = [INF for u in range(V)]               # INF = 1e9 here
dist[s] = 0
for i in range(0, V-1):                      # total O(V*E)
    modified = False                         # optimization
    for u in range(0, V):                    # these two loops = O(E)
        if (not dist[u] == INF):             # important check
            for v, w in AL[u]:
                if (dist[u]+w >= dist[v]): continue # not improving, skip
                dist[v] = dist[u]+w          # relax operation
                modified = True              # optimization
    if (not modified): break                 # optimization
    
hasNegativeCycle = False
for u in range(0, V):                        # one more pass to check negative cycle
    if (not dist[u] == INF):
        for v, w in AL[u]:
            if (dist[v] > dist[u] + w):      # should be false
                hasNegativeCycle = True      # if true => -ve cycle
```
#### SPFA

```python
# SPFA from source S
# initially, only source vertex s has dist[s] = 0 and in the queue
from collections import deque
INF = int(1E9)
s = 0 # start vertex
dist = [INF for u in range(V)]
dist[s] = 0
q = deque()
q.append(s)
in_queue = [0 for u in range(V)]
in_queue[s] = 1
while (len(q) > 0):
    u = q.popleft()                          # pop from queue
    in_queue[u] = 0
    for v, w in AL[u]:
        if (dist[u]+w >= dist[v]): continue  # not improving, skip
        dist[v] = dist[u]+w                  # relax operation
        if not in_queue[v]:                  # add to the queue
            q.append(v)                      # only if v is not
            in_queue[v] = 1                  # already in the queue
```

#### Dijkstra

```python
from heapq import heappush, heappop
# (Modified) Dijkstra's routine #(V+E) log V
dist = [INF for u in range(V)]
dist[s] = 0 # s = start vertex
pq = []
heappush(pq, (0, s))
# sort the pairs by non-decreasing distance from s
while (len(pq) > 0):                    # main loop
    d, u = heappop(pq)                  # shortest unvisited u
    if (d > dist[u]): continue          # a very important check
    for v, w in AL[u]:                  # all edges from u
        if (dist[u]+w >= dist[v]): continue # not improving, skip
        dist[v] = dist[u]+w             # relax operation
        heappush(pq, (dist[v], v))
```


## Special Problems

### Big Number
![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312070156606.png)
![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312070156635.png)

### Sort 
![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312070123487.png)

### Stack and Queue

#### bracketmatching
```python
stack = []
n = int(input())
s = input()
idx = 0
flag = True
d = {"}":"{", ")":"(", "]":"["}
while n:
	n-=1
	if s[idx] in set("{(["):
		stack.append(s[idx])
		idx += 1
		continue
	if s[idx] in set(")}]"):
		if len(stack) == 0 or stack[-1] != d[s[idx]]:
			flag = False
			break
		else:
			stack.pop()
		idx += 1
print("Valid") if flag and not len(stack) else print("Invalid")
```

#### Rational Sequence

![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312071322199.png)


```python
for _ in range(int(input())):
    K, N = map(int, input().split())
    st = []
    while N > 1:
        if  N % 2 == 1: st.append("R")
        else:           st.append("L")
        N //= 2
    p, q = 1, 1
    while st:
        cmd = st.pop()
        if cmd == "L": q = (p+q)
        else:          p = (p+q)
    print(K," ",p,'/',q,sep="")
```
### Graph

![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312070146108.png)
![image.png](https://cdn.jsdelivr.net/gh/ayhhyhh/IMGbed@master/imgs/202312070146109.png)

