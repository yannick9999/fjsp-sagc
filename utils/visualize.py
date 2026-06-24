import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x)


def plot_operation_graph(
    adj,
    opes_appertain,
    nums_opes,
    labels=None,
    title="Operation Graph",
    save_path=None,
    show=True,
    transpose_adj=False,
    ope_ma_adj=None,
    show_machines=False,
    eligible_opes=None,
    completed_opes=None,
    running_opes=None,
):
    adj = _to_numpy(adj)
    opes_appertain = _to_numpy(opes_appertain)
    labels = _to_numpy(labels)
    ope_ma_adj = _to_numpy(ope_ma_adj)
    eligible_opes = _to_numpy(eligible_opes)
    completed_opes = _to_numpy(completed_opes)
    running_opes = _to_numpy(running_opes)

    G = nx.DiGraph()
    for i in range(nums_opes):
        G.add_node(i)

    op_edges = []
    for i in range(nums_opes):
        for j in range(nums_opes):
            if adj[i, j] > 0:
                if transpose_adj:
                    op_edges.append((j, i))
                else:
                    op_edges.append((i, j))
    G.add_edges_from(op_edges)

    num_mas = 0
    ma_edges = []
    if show_machines and ope_ma_adj is not None:
        num_mas = ope_ma_adj.shape[1]
        for m in range(num_mas):
            G.add_node(f"M{m}")
        for i in range(nums_opes):
            for m in range(num_mas):
                if ope_ma_adj[i, m] > 0:
                    ma_edges.append((i, f"M{m}"))

    pos = {}
    job_op_counter = {}
    for i in range(nums_opes):
        job = int(opes_appertain[i])
        x = job_op_counter.get(job, 0)
        y = -job
        pos[i] = (x, y)
        job_op_counter[job] = x + 1

    max_x = max(v[0] for v in pos.values()) if pos else 0

    if show_machines and num_mas > 0:
        spacing = max_x / (num_mas - 1) if num_mas > 1 else 0
        x_offset = (max_x - spacing * (num_mas - 1)) / 2 if num_mas > 1 else max_x / 2
        half_step = spacing / 2 if num_mas > 1 else 0.5
        for m in range(num_mas):
            pos[f"M{m}"] = (x_offset + m * spacing + half_step, 1.5)

    num_jobs = int(opes_appertain[:nums_opes].max()) + 1
    cmap = cm.tab10 if num_jobs <= 10 else cm.tab20
    op_colors = [cmap(int(opes_appertain[i]) % cmap.N) for i in range(nums_opes)]
    op_nodes = list(range(nums_opes))

    edgecolors = []
    linewidths = []
    for i in range(nums_opes):
        if eligible_opes is not None and eligible_opes[i]:
            edgecolors.append('green')
            linewidths.append(3.0)
        elif running_opes is not None and running_opes[i]:
            edgecolors.append('orange')
            linewidths.append(3.0)
        elif completed_opes is not None and completed_opes[i]:
            edgecolors.append('grey')
            linewidths.append(2.0)
        else:
            edgecolors.append('black')
            linewidths.append(1.0)

    node_labels = {}
    for i in range(nums_opes):
        node_labels[i] = str(int(labels[i])) if labels is not None else str(i)

    fig_h = max(4, num_jobs * 1.2 + 1)
    if show_machines and num_mas > 0:
        fig_h += 1.5
    fig, ax = plt.subplots(figsize=(max(6, max_x * 1.5 + 2), fig_h))

    nx.draw_networkx_nodes(G, pos, nodelist=op_nodes, node_color=op_colors,
                           node_size=500, edgecolors=edgecolors,
                           linewidths=linewidths, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=op_edges, arrows=True,
                           arrowstyle='->', arrowsize=15,
                           connectionstyle='arc3,rad=0.05', ax=ax)

    if show_machines and num_mas > 0:
        ma_nodes = [f"M{m}" for m in range(num_mas)]
        ma_labels = {f"M{m}": f"M{m}" for m in range(num_mas)}
        nx.draw_networkx_nodes(G, pos, nodelist=ma_nodes, node_color='lightgrey',
                               node_shape='s', node_size=400, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=ma_labels, font_size=7, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=ma_edges, arrows=False,
                               style='dashed', edge_color='grey', alpha=0.5, ax=ax)

    has_status = any(x is not None for x in [eligible_opes, completed_opes, running_opes])
    if has_status:
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                   markeredgewidth=1.0, markersize=10, label='Unscheduled'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='green',
                   markeredgewidth=3.0, markersize=10, label='Eligible'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='orange',
                   markeredgewidth=3.0, markersize=10, label='Running'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='grey',
                   markeredgewidth=2.0, markersize=10, label='Done'),
        ]
        ax.legend(handles=legend_items, loc='lower right', fontsize=7, framealpha=0.8)

    ax.set_title(title)
    ax.margins(0.15)
    ax.axis('off')
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
