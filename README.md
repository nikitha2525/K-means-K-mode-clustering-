# 🔵 Day 38 — K-Means++ & K-Modes Clustering

> **100 Days AI/ML Engineer Challenge**  
> Fixing K-Means weaknesses and extending clustering to categorical data — built from scratch.

---

## 📌 What This Covers

| Topic | Concept |
|---|---|
| K-Means++ | Better centroid initialization via distance probability |
| K-Modes | Categorical clustering using mismatch distance |
| From-scratch | No `sklearn.KMeans()` — full manual implementation |

---

## 🧠 The Problem with Vanilla K-Means

Random initialization → centroids can cluster near each other → poor separation → bad results.

```
Cluster 1 ●●●  ←  centroid
Cluster 2 ●●●  ←  centroid (accidentally nearby)
Cluster 3 ●●●     (not even assigned its own centroid!)
```

K-Means++ fixes this by making initialization **probabilistic** — farther points get higher selection weight.

---

## 🔵 K-Means++ — Improved Initialization

### The Core Formula

$$P(x) = \frac{D(x)^2}{\sum D(x)^2}$$

Where `D(x)` = distance from point `x` to its nearest already-chosen centroid.

**In plain English:** the farther a point is from existing centroids, the more likely it is to become the next one. This guarantees **spread**.

### Implementation Walkthrough

```python
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans_plus_plus_init(X, k):
    """
    K-Means++ centroid initialization.
    Returns k centroids chosen with distance-weighted probability.
    """
    centroids = []

    # Step 1: First centroid — purely random
    first_idx = np.random.randint(0, len(X))
    centroids.append(X[first_idx])

    # Steps 2..k: Probabilistic selection
    for _ in range(k - 1):
        # D(x)^2 for each point — distance to nearest centroid
        distances = np.array([
            min(euclidean_distance(x, c) ** 2 for c in centroids)
            for x in X
        ])

        # Normalize to get probabilities
        probabilities = distances / distances.sum()

        # Sample next centroid based on probability weights
        next_idx = np.random.choice(len(X), p=probabilities)
        centroids.append(X[next_idx])

    return np.array(centroids)
```

### Generating Test Data (4 Gaussian Clusters)

```python
np.random.seed(42)

cluster_1 = np.random.multivariate_normal([1, 1],   [[0.5, 0], [0, 0.5]], 100)
cluster_2 = np.random.multivariate_normal([5, 5],   [[0.5, 0], [0, 0.5]], 100)
cluster_3 = np.random.multivariate_normal([1, 5],   [[0.5, 0], [0, 0.5]], 100)
cluster_4 = np.random.multivariate_normal([5, 1],   [[0.5, 0], [0, 0.5]], 100)

X = np.vstack([cluster_1, cluster_2, cluster_3, cluster_4])
```

### Full K-Means Loop

```python
def kmeans(X, k, max_iters=100):
    # K-Means++ initialization
    centroids = kmeans_plus_plus_init(X, k)

    for iteration in range(max_iters):
        # Assignment step: each point → nearest centroid
        labels = np.array([
            np.argmin([euclidean_distance(x, c) for c in centroids])
            for x in X
        ])

        # Update step: recompute centroids as cluster means
        new_centroids = np.array([
            X[labels == i].mean(axis=0) for i in range(k)
        ])

        # Convergence check
        if np.allclose(centroids, new_centroids):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return labels, centroids

labels, centroids = kmeans(X, k=4)
```

### Visualization

```python
import matplotlib.pyplot as plt

colors = ['#4f8eff', '#22d3a8', '#f59e0b', '#f87171']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: K-Means++ initialization step
init_centroids = kmeans_plus_plus_init(X, k=4)
axes[0].scatter(X[:, 0], X[:, 1], c='#374151', s=20, alpha=0.5)
axes[0].scatter(init_centroids[:-1, 0], init_centroids[:-1, 1],
                c='black', s=150, marker='x', linewidths=2.5, label='Selected centroids')
axes[0].scatter(init_centroids[-1, 0], init_centroids[-1, 1],
                c='red', s=200, marker='*', label='Next centroid (prob-weighted)')
axes[0].set_title('K-Means++ Initialization', fontweight='bold')
axes[0].legend(fontsize=9)

# Right: Final clusters
for i in range(4):
    mask = labels == i
    axes[1].scatter(X[mask, 0], X[mask, 1], c=colors[i], s=20, alpha=0.7, label=f'Cluster {i+1}')
axes[1].scatter(centroids[:, 0], centroids[:, 1],
                c='white', s=200, marker='X', edgecolors='black', linewidths=1.5, label='Centroids')
axes[1].set_title('Final K-Means++ Clusters', fontweight='bold')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('kmeans_plus_plus.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔴 K-Modes — Categorical Clustering

### Why K-Means fails on categories

```python
# K-Means uses mean — meaningless for strings:
np.mean(['Blue', 'Red', 'Black'])   # ← TypeError. Breaks immediately.

# Euclidean distance — meaningless for categories:
distance('Blue', 'Red')             # ← What does this even mean?
```

K-Modes replaces both with category-friendly alternatives:

| K-Means | K-Modes |
|---|---|
| Centroid = **mean** | Centroid = **mode** (most frequent value) |
| Distance = **Euclidean** | Distance = **mismatch count** |

### Mismatch Distance

```python
def hamming_distance(point, mode):
    """Count of attributes where point != mode."""
    return np.sum(point != mode)
```

### Sample Categorical Dataset

```python
import pandas as pd

data = np.array([
    ['Blue',   'Large',  'Circle'],
    ['Red',    'Small',  'Square'],
    ['Blue',   'Large',  'Circle'],
    ['Black',  'Medium', 'Triangle'],
    ['Red',    'Small',  'Square'],
    ['Black',  'Medium', 'Triangle'],
    ['Blue',   'Small',  'Circle'],
    ['Red',    'Large',  'Square'],
    ['Black',  'Large',  'Triangle'],
])
```

### Full K-Modes Implementation

```python
def kmodes(data, k, max_iters=100):
    n, n_features = data.shape

    # Step 1: Random initialization — pick k rows as initial modes
    init_indices = np.random.choice(n, k, replace=False)
    modes = data[init_indices].copy()

    for iteration in range(max_iters):
        # Assignment: each point → mode with fewest mismatches
        labels = np.array([
            np.argmin([np.sum(point != mode) for mode in modes])
            for point in data
        ])

        # Update: recompute mode (most frequent value per feature per cluster)
        new_modes = np.array([
            pd.DataFrame(data[labels == i]).mode().iloc[0].values
            for i in range(k)
        ])

        # Convergence check
        if np.array_equal(modes, new_modes):
            print(f"  Converged at iteration {iteration + 1}")
            break

        modes = new_modes

    return labels, modes


labels, modes = kmodes(data, k=3)

print("Cluster Assignments:", labels)
print("\nFinal Modes (cluster centers):")
for i, mode in enumerate(modes):
    print(f"  Cluster {i}: {mode}")
```

### Sample Output

```
Converged at iteration 3

Cluster Assignments: [0 1 0 2 1 2 0 1 2]

Final Modes (cluster centers):
  Cluster 0: ['Blue' 'Large' 'Circle']
  Cluster 1: ['Red' 'Small' 'Square']
  Cluster 2: ['Black' 'Medium' 'Triangle']
```

---

## ⚖️ K-Means vs K-Means++ vs K-Modes

| Feature | K-Means | K-Means++ | K-Modes |
|---|---|---|---|
| Data type | Numerical | Numerical | **Categorical** |
| Initialization | Random | Distance-weighted | Random |
| Centroid type | Mean | Mean | **Mode** |
| Distance metric | Euclidean | Euclidean | **Mismatch count** |
| Convergence speed | Slower | **Faster** | Medium |
| Cluster shape | Spherical | Spherical | Any |

---

## ⚠️ Known Limitations

- **K-Means++** still assumes spherical, similarly-sized clusters
- **K-Modes** is sensitive to initialization (no ++ variant used here)
- Both require choosing `k` upfront — use the elbow method or silhouette score to guide it
- Neither handles mixed data (numerical + categorical) — for that, look at **K-Prototypes**

---

## 🔑 Key Takeaways

```
"If you don't understand how centroids are initialized,
 you don't understand K-Means."
```

1. **Initialization is not a minor detail** — it decides convergence quality
2. **Distance metric defines the algorithm** — swap Euclidean for mismatch, you get a completely different clustering behavior
3. **K-Means is not universal** — categorical data needs a categorically different approach
4. **Building from scratch > calling fit()** — you now understand what `sklearn` is doing internally

---

## 📁 Files

```
day38/
├── README.md                  ← this file
├── kmeans_plus_plus.py        ← K-Means++ from scratch
├── kmodes.py                  ← K-Modes from scratch
└── kmeans_plus_plus.png       ← initialization + cluster visualization
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

[![LinkedIn](https://img.shields.io/badge/Follow_the_journey-LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/)

`#Day38` `#KMeansPlusPlus` `#KModes` `#Clustering` `#UnsupervisedLearning` `#100DaysAIML`
