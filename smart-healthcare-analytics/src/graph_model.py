import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os

def create_feature_graph(df: pd.DataFrame, corr_threshold=0.5):
    """
    Create a graph where nodes are features and edges connect
    features with strong correlation.
    """
    corr = df.corr(numeric_only=True)
    G = nx.Graph()

    # Add nodes (features)
    for feature in corr.columns:
        G.add_node(feature)

    # Add edges where correlation exceeds threshold
    for f1 in corr.columns:
        for f2 in corr.columns:
            if f1 != f2 and abs(corr.loc[f1, f2]) >= corr_threshold:
                G.add_edge(f1, f2, weight=round(corr.loc[f1, f2], 2))

    return G

def analyze_graph(G: nx.Graph):
    """
    Compute key graph metrics: degree, centrality, and clustering.
    """
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 2),
        "avg_clustering": round(nx.average_clustering(G), 2),
        "degree_centrality": {n: round(v, 2) for n, v in nx.degree_centrality(G).items()}
    }
    return metrics

def visualize_graph(G: nx.Graph, output_path="docs/plots/feature_graph.png"):
    """
    Visualize graph and save to file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1200, node_color="lightblue", edge_color="gray", font_size=9)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Feature Correlation Graph")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
