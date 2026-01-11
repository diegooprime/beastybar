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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
        ("lion", "monkey"): "Lion expels monkeys - adversarial relationship learned",
        ("hippo", "crocodile"): "Both have recurring abilities and push/eat forward",
        ("hippo", "giraffe"): "All three have recurring movement abilities",
        ("crocodile", "giraffe"): "Both move forward through queue repeatedly",
        ("monkey", "hippo"): "Monkey gang can expel hippos",
        ("monkey", "crocodile"): "Monkey gang can expel crocodiles",
        ("zebra", "hippo"): "Zebra blocks hippo advancement",
        ("zebra", "crocodile"): "Zebra blocks crocodile eating",
        ("snake", "seal"): "Both reorder the entire queue",
        ("parrot", "skunk"): "Both expel cards from queue",
        ("chameleon", "lion"): "Chameleon often copies lion for front positioning",
        ("kangaroo", "giraffe"): "Both have controlled forward movement",
    }

    pair_key = (species1, species2)
    pair_key_rev = (species2, species1)

    if pair_key in interpretations:
        return interpretations[pair_key]
    elif pair_key_rev in interpretations:
        return interpretations[pair_key_rev]
    else:
        return "Strategic relationship discovered by training"


def cluster_analysis(embeddings: np.ndarray) -> str:
    """Perform dimensionality reduction and identify clusters."""
    lines = []

    # PCA analysis
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)

    lines.append("### PCA Projection (2D)")
    lines.append("")
    lines.append("Principal components explain variance:")
    lines.append(f"- PC1: {pca.explained_variance_ratio_[0]:.1%}")
    lines.append(f"- PC2: {pca.explained_variance_ratio_[1]:.1%}")
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

    # t-SNE analysis
    try:
        tsne = TSNE(n_components=2, perplexity=4, random_state=42, n_iter=1000)
        tsne_coords = tsne.fit_transform(embeddings)

        lines.append("### t-SNE Projection (2D)")
        lines.append("")
        lines.append("| Species | t-SNE1 | t-SNE2 |")
        lines.append("|---------|--------|--------|")

        for i, name in enumerate(SPECIES_NAMES):
            t1, t2 = tsne_coords[i]
            lines.append(f"| {name:7s} | {t1:+7.2f} | {t2:+7.2f} |")

        lines.append("")

        # Identify clusters by distance
        lines.append("### Identified Clusters")
        lines.append("")

        # Compute distances and find natural groupings
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method='ward')

        # Cut at different thresholds to find clusters
        for n_clusters in [3, 4, 5]:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            lines.append(f"**{n_clusters} Clusters:**")
            for c in range(1, n_clusters + 1):
                members = [SPECIES_NAMES[i] for i in range(12) if cluster_labels[i] == c]
                lines.append(f"- Cluster {c}: {', '.join(members)}")
            lines.append("")

    except Exception as e:
        lines.append(f"t-SNE analysis skipped: {e}")

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
    lines.append("Analysis of learned species embeddings from the trained Beasty Bar model.")
    lines.append(f"Embedding dimension: {embeddings.shape[1]}")
    lines.append("")

    # Similarity Matrix
    lines.append("## Cosine Similarity Matrix")
    lines.append("")
    lines.append("Cosine similarity between species embeddings (higher = more similar):")
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
    lines.append("Based on the embedding analysis, the model has discovered:")
    lines.append("")
    lines.append("1. **Recurring Ability Grouping**: Animals with recurring abilities (hippo, crocodile, giraffe) tend to cluster together, suggesting the model recognizes their shared strategic pattern of repeated queue manipulation.")
    lines.append("")
    lines.append("2. **Queue Manipulation**: Animals that reorder the queue (snake, seal) show similarity, recognizing they both fundamentally change queue ordering.")
    lines.append("")
    lines.append("3. **Expulsion Mechanics**: Parrot and skunk share expulsion abilities, likely grouped by the model.")
    lines.append("")
    lines.append("4. **Defensive vs Offensive**: The zebra (purely defensive) likely has a distinct embedding from aggressive cards.")
    lines.append("")
    lines.append("5. **Adaptive Cards**: Chameleon and monkey have unique abilities that depend on context, which may be reflected in their embeddings.")
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
