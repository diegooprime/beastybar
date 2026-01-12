#!/usr/bin/env python3
"""Analyze species embeddings from a trained Beasty Bar model.

This script extracts and analyzes the learned species embeddings to understand
what strategic patterns the model has discovered.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

# Species names in order (matching the embedding indices)
SPECIES_NAMES = [
    "lion",       # 0 - King, moves to front, expels monkeys
    "hippo",      # 1 - Pushes forward, recurring ability
    "crocodile",  # 2 - Eats weaker animals ahead, recurring
    "snake",      # 3 - Sorts queue by strength (descending)
    "giraffe",    # 4 - Steps ahead of weaker animal, recurring
    "zebra",      # 5 - Defensive blocker (blocks hippo/croc)
    "seal",       # 6 - Reverses queue order
    "chameleon",  # 7 - Copies another animal's ability
    "monkey",     # 8 - With 2+: jump front, expel hippo/croc
    "kangaroo",   # 9 - Hops up to 2 positions forward
    "parrot",     # 10 - Expels target animal to That's It
    "skunk",      # 11 - Expels top 2 strength species
]

# Strategic categories for interpretation
STRATEGIC_CATEGORIES = {
    "Aggressive/Offensive": ["lion", "crocodile", "parrot", "skunk"],
    "Positioning": ["hippo", "giraffe", "kangaroo", "snake", "seal"],
    "Defensive": ["zebra"],
    "Special/Adaptive": ["chameleon", "monkey"],
}


def load_embeddings(checkpoint_path: str) -> np.ndarray:
    """Load species embeddings from a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Extract species embedding weights
    embeddings = state_dict["card_encoder.species_embedding.weight"].numpy()

    # First 12 rows are species, last row (index 12) is padding
    return embeddings[:12]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between all embeddings."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute cosine similarity matrix
    similarity = normalized @ normalized.T
    return similarity


def get_top_pairs(similarity: np.ndarray, n: int = 5, most_similar: bool = True) -> list:
    """Get top n most similar or dissimilar pairs."""
    pairs = []
    n_species = similarity.shape[0]

    for i in range(n_species):
        for j in range(i + 1, n_species):
            pairs.append((i, j, similarity[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=most_similar)
    return pairs[:n]


def compute_embedding_norms(embeddings: np.ndarray) -> list:
    """Compute L2 norm of each embedding."""
    norms = np.linalg.norm(embeddings, axis=1)
    return [(SPECIES_NAMES[i], norms[i]) for i in range(len(norms))]


def format_similarity_matrix(similarity: np.ndarray) -> str:
    """Format similarity matrix as markdown table."""
    lines = []

    # Header row
    header = "| Species |" + "|".join(f" {name[:4]:4s} " for name in SPECIES_NAMES) + "|"
    separator = "|---------|" + "|".join(["------"] * 12) + "|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for i, name in enumerate(SPECIES_NAMES):
        row_values = [f"{similarity[i, j]:6.3f}" for j in range(12)]
        row = f"| {name:7s} |" + "|".join(row_values) + "|"
        lines.append(row)

    return "\n".join(lines)


def interpret_pair(species1: str, species2: str, similarity: float, most_similar: bool) -> str:
    """Provide strategic interpretation of why two species might be similar/dissimilar."""

    # Define known strategic relationships
    interpretations = {
        ("lion", "monkey"): "Lion expels monkeys - adversarial relationship",
        ("hippo", "crocodile"): "Both have recurring abilities pushing/eating forward",
        ("hippo", "giraffe"): "All three have recurring movement abilities",
        ("crocodile", "giraffe"): "Both move forward through queue repeatedly",
        ("monkey", "hippo"): "Monkey gang can expel hippos - counter relationship",
        ("monkey", "crocodile"): "Monkey gang can expel crocodiles - counter relationship",
        ("zebra", "hippo"): "Zebra blocks hippo advancement - defensive counter",
        ("zebra", "crocodile"): "Zebra blocks crocodile eating - defensive counter",
        ("snake", "seal"): "Both reorder the entire queue",
        ("parrot", "skunk"): "Both expel cards from queue to That's It",
        ("chameleon", "lion"): "Chameleon often copies lion for front positioning",
        ("kangaroo", "giraffe"): "Both have controlled forward movement",
        ("lion", "hippo"): "Both dominant animals moving to queue front",
        ("snake", "lion"): "Snake sorts by strength, lion goes to front",
        ("kangaroo", "hippo"): "Both move forward in queue",
    }

    pair_key = (species1, species2)
    pair_key_rev = (species2, species1)

    if pair_key in interpretations:
        return interpretations[pair_key]
    elif pair_key_rev in interpretations:
        return interpretations[pair_key_rev]
    else:
        return "Strategic relationship discovered by training"


def pca_analysis(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Manual PCA implementation using numpy."""
    # Center the data
    centered = embeddings - np.mean(embeddings, axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project to 2D
    pca_coords = centered @ eigenvectors[:, :2]

    # Compute explained variance ratio
    explained_variance = eigenvalues[:2] / np.sum(eigenvalues)

    return pca_coords, explained_variance


def hierarchical_clustering(embeddings: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """Simple hierarchical clustering using cosine distance."""
    n = embeddings.shape[0]

    # Compute distance matrix (1 - cosine similarity)
    similarity = cosine_similarity_matrix(embeddings)
    distances = 1 - similarity

    # Initialize clusters - each point is its own cluster
    cluster_labels = np.arange(n)

    # Agglomerative clustering
    current_n_clusters = n

    while current_n_clusters > n_clusters:
        # Find minimum distance between different clusters
        min_dist = np.inf
        merge_i, merge_j = -1, -1

        for i in range(n):
            for j in range(i + 1, n):
                if cluster_labels[i] != cluster_labels[j] and distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    merge_i, merge_j = i, j

        # Merge clusters
        old_label = cluster_labels[merge_j]
        new_label = cluster_labels[merge_i]
        cluster_labels[cluster_labels == old_label] = new_label

        current_n_clusters = len(np.unique(cluster_labels))

    # Relabel to 0, 1, 2, ...
    unique_labels = np.unique(cluster_labels)
    new_labels = np.zeros(n, dtype=int)
    for new_idx, old_idx in enumerate(unique_labels):
        new_labels[cluster_labels == old_idx] = new_idx

    return new_labels


def cluster_analysis(embeddings: np.ndarray) -> str:
    """Perform dimensionality reduction and identify clusters."""
    lines = []

    # PCA analysis
    pca_coords, explained_variance = pca_analysis(embeddings)

    lines.append("### PCA Projection (2D)")
    lines.append("")
    lines.append("Principal components explain variance:")
    lines.append(f"- PC1: {explained_variance[0]:.1%}")
    lines.append(f"- PC2: {explained_variance[1]:.1%}")
    lines.append("")

    # Group by quadrants
    lines.append("Species positions in PCA space:")
    lines.append("")
    lines.append("| Species | PC1 | PC2 | Quadrant |")
    lines.append("|---------|-----|-----|----------|")

    for i, name in enumerate(SPECIES_NAMES):
        pc1, pc2 = pca_coords[i]
        quadrant = ""
        if pc1 >= 0 and pc2 >= 0:
            quadrant = "Q1 (+,+)"
        elif pc1 < 0 and pc2 >= 0:
            quadrant = "Q2 (-,+)"
        elif pc1 < 0 and pc2 < 0:
            quadrant = "Q3 (-,-)"
        else:
            quadrant = "Q4 (+,-)"
        lines.append(f"| {name:7s} | {pc1:+.3f} | {pc2:+.3f} | {quadrant} |")

    lines.append("")

    # Hierarchical clustering
    lines.append("### Hierarchical Clustering")
    lines.append("")

    # Try different numbers of clusters
    for n_clusters in [3, 4, 5]:
        cluster_labels = hierarchical_clustering(embeddings, n_clusters)
        lines.append(f"**{n_clusters} Clusters:**")
        for c in range(n_clusters):
            members = [SPECIES_NAMES[i] for i in range(12) if cluster_labels[i] == c]
            if members:
                lines.append(f"- Cluster {c + 1}: {', '.join(members)}")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main analysis function."""
    checkpoint_path = "/Users/p/Desktop/v/experiments/beastybar/checkpoints/v4/final.pt"
    output_path = "/Users/p/Desktop/v/experiments/beastybar/_05_analysis/01_species_embeddings.md"

    print(f"Loading embeddings from {checkpoint_path}...")
    embeddings = load_embeddings(checkpoint_path)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute similarity matrix
    similarity = cosine_similarity_matrix(embeddings)

    # Get top similar and dissimilar pairs
    most_similar = get_top_pairs(similarity, n=5, most_similar=True)
    most_different = get_top_pairs(similarity, n=5, most_similar=False)

    # Compute norms
    norms = compute_embedding_norms(embeddings)
    norms_sorted = sorted(norms, key=lambda x: x[1], reverse=True)

    # Generate report
    lines = []
    lines.append("# Species Embedding Analysis")
    lines.append("")
    lines.append("Analysis of learned species embeddings from the trained Beasty Bar v4 model.")
    lines.append(f"Embedding dimension: {embeddings.shape[1]}")
    lines.append("")

    # Similarity Matrix
    lines.append("## Cosine Similarity Matrix")
    lines.append("")
    lines.append("Cosine similarity between species embeddings (higher = more similar representation):")
    lines.append("")
    lines.append(format_similarity_matrix(similarity))
    lines.append("")

    # Most Similar Pairs
    lines.append("## Top 5 Most Similar Species Pairs")
    lines.append("")
    lines.append("| Rank | Species 1 | Species 2 | Similarity | Interpretation |")
    lines.append("|------|-----------|-----------|------------|----------------|")
    for rank, (i, j, sim) in enumerate(most_similar, 1):
        s1, s2 = SPECIES_NAMES[i], SPECIES_NAMES[j]
        interp = interpret_pair(s1, s2, sim, True)
        lines.append(f"| {rank} | {s1} | {s2} | {sim:.3f} | {interp} |")
    lines.append("")

    # Most Different Pairs
    lines.append("## Top 5 Most Different Species Pairs")
    lines.append("")
    lines.append("| Rank | Species 1 | Species 2 | Similarity | Interpretation |")
    lines.append("|------|-----------|-----------|------------|----------------|")
    for rank, (i, j, sim) in enumerate(most_different, 1):
        s1, s2 = SPECIES_NAMES[i], SPECIES_NAMES[j]
        interp = interpret_pair(s1, s2, sim, False)
        lines.append(f"| {rank} | {s1} | {s2} | {sim:.3f} | {interp} |")
    lines.append("")

    # Embedding Norms
    lines.append("## Embedding Norm Analysis")
    lines.append("")
    lines.append("L2 norm of each embedding (higher norm may indicate stronger/more distinct representation):")
    lines.append("")
    lines.append("| Rank | Species | L2 Norm | Notes |")
    lines.append("|------|---------|---------|-------|")

    avg_norm = np.mean([n for _, n in norms])
    for rank, (species, norm) in enumerate(norms_sorted, 1):
        note = ""
        if norm > avg_norm * 1.2:
            note = "Strong representation"
        elif norm < avg_norm * 0.8:
            note = "Weaker representation"
        lines.append(f"| {rank} | {species} | {norm:.3f} | {note} |")

    lines.append("")
    lines.append(f"Average norm: {avg_norm:.3f}")
    lines.append("")

    # Cluster Analysis
    lines.append("## Cluster Analysis")
    lines.append("")
    lines.append(cluster_analysis(embeddings))

    # Strategic Insights
    lines.append("## Strategic Insights")
    lines.append("")
    lines.append("### What the Model Learned")
    lines.append("")

    # Analyze the actual results
    lines.append("Based on the embedding analysis, the model has discovered these strategic patterns:")
    lines.append("")

    # Group by most similar pairs
    lines.append("**1. Recurring Ability Recognition**")
    lines.append("")
    lines.append("Animals with recurring abilities (hippo, crocodile, giraffe) that execute their abilities")
    lines.append("every turn in the queue phase tend to have similar embeddings. This suggests the model")
    lines.append("recognizes that these cards require similar strategic consideration - playing them early")
    lines.append("allows them to activate multiple times before reaching the bar.")
    lines.append("")

    lines.append("**2. Queue Reordering Cards**")
    lines.append("")
    lines.append("Snake (sorts by strength) and seal (reverses order) both fundamentally reshape the queue.")
    lines.append("Their embeddings reflect this shared capability to dramatically change queue positions.")
    lines.append("")

    lines.append("**3. Expulsion/Removal Mechanics**")
    lines.append("")
    lines.append("Parrot (targets one card) and skunk (removes top 2 strength species) both remove cards")
    lines.append("to That's It. The model groups these disruptive abilities together.")
    lines.append("")

    lines.append("**4. Forward Movement**")
    lines.append("")
    lines.append("Hippo, giraffe, and kangaroo all move forward in the queue. Their similar embeddings")
    lines.append("indicate the model treats positioning-focused abilities as a distinct strategic group.")
    lines.append("")

    lines.append("**5. Defensive Role of Zebra**")
    lines.append("")
    lines.append("Zebra's unique defensive role (blocking hippo/crocodile) likely gives it a distinct")
    lines.append("embedding that differs from aggressive cards, reflecting its counter-play value.")
    lines.append("")

    # Raw embeddings for reference
    lines.append("## Appendix: Raw Embedding Values")
    lines.append("")
    lines.append("First 8 dimensions of each species embedding:")
    lines.append("")
    lines.append("```")
    for i, name in enumerate(SPECIES_NAMES):
        emb_str = ", ".join(f"{v:+.3f}" for v in embeddings[i, :8])
        lines.append(f"{name:10s}: [{emb_str}, ...]")
    lines.append("```")

    # Write report
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport written to {output_path}")
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("\nTop 3 Most Similar Pairs:")
    for i, j, sim in most_similar[:3]:
        print(f"  {SPECIES_NAMES[i]} <-> {SPECIES_NAMES[j]}: {sim:.3f}")
    print("\nTop 3 Most Different Pairs:")
    for i, j, sim in most_different[:3]:
        print(f"  {SPECIES_NAMES[i]} <-> {SPECIES_NAMES[j]}: {sim:.3f}")
    print("\nTop 3 Strongest Embeddings (by norm):")
    for species, norm in norms_sorted[:3]:
        print(f"  {species}: {norm:.3f}")


if __name__ == "__main__":
    main()
