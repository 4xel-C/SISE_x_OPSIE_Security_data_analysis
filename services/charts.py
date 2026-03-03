"""
Chart service for security log visualization.
Each function takes a DataFrame (aggregated by ipsrc) and returns a Plotly Figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from pandas import DataFrame
import numpy as np

if TYPE_CHECKING:
    from services.clustering_service import ClusteringResult


# =============================================================================
# ANALYSE DESCRIPTIVE (raw logs)
# =============================================================================


def proto_action_bar(df_raw: DataFrame) -> go.Figure:
    """Grouped bar chart of Permit/Deny counts per protocol (TCP, UDP).
    Works on the raw log DataFrame (one row per flow).
    """
    rows = []
    for proto in ["TCP", "UDP"]:
        sub = df_raw[df_raw["proto"] == proto]
        rows.append({
            "Protocole": proto,
            "Permit": int((sub["action"] == "Permit").sum()),
            "Deny": int((sub["action"] == "Deny").sum()),
        })
    plot_df = DataFrame(rows)
    fig = px.bar(
        plot_df,
        x="Protocole",
        y=["Permit", "Deny"],
        barmode="group",
        title="Flux par protocole & action",
        color_discrete_map={"Permit": "#34d399", "Deny": "#f87171"},
        labels={"value": "Nombre de flux", "variable": "Action"},
    )
    return fig


def allow_deny_pie(df_raw: DataFrame) -> go.Figure:
    """Donut chart of Permit / Deny distribution from raw logs."""
    permit = int((df_raw["action"] == "Permit").sum())
    deny = int((df_raw["action"] == "Deny").sum())
    fig = go.Figure(go.Pie(
        labels=["Permit", "Deny"],
        values=[permit, deny],
        hole=0.45,
        marker_colors=["#34d399", "#f87171"],
    ))
    fig.update_layout(title="Répartition Permit / Deny")
    return fig


def port_distribution_bar(df_raw: DataFrame) -> go.Figure:
    """Horizontal stacked bar chart of flows by destination port range (TCP vs UDP).
    The last bucket explicitly highlights client/ephemeral ports (49152–65535).
    """
    buckets = [
        ("0–1023", 0, 1023),
        ("1024–5000", 1024, 5000),
        ("5001–10000", 5001, 10000),
        ("10001–49151", 10001, 49151),
        ("Clients (49152–65535)", 49152, 65535),
    ]
    rows = []
    for name, lo, hi in buckets:
        sub = df_raw[(df_raw["portdst"] >= lo) & (df_raw["portdst"] <= hi)]
        rows.append({
            "Plage": name,
            "TCP": int((sub["proto"] == "TCP").sum()),
            "UDP": int((sub["proto"] == "UDP").sum()),
        })
    plot_df = DataFrame(rows)
    fig = px.bar(
        plot_df,
        x=["TCP", "UDP"],
        y="Plage",
        orientation="h",
        barmode="stack",
        title="Distribution par plages de ports destination",
        color_discrete_map={"TCP": "#818cf8", "UDP": "#fbbf24"},
        labels={"value": "Nombre de flux", "variable": "Protocole"},
    )
    return fig


def ip_flux_vs_dest_scatter(df_raw: DataFrame) -> go.Figure:
    """Scatter plot of IP sources: total flux (x) vs unique destinations (y).
    Bubble size = total flux, color = categorical ALLOW/DENY majority.
    custom_data[0] = ipsrc for click-event identification.
    """
    agg = df_raw.groupby("ipsrc").agg(
        total_flux=("ipsrc", "count"),
        distinct_dst=("ipdst", "nunique"),
        permit=("action", lambda x: (x == "Permit").sum()),
        deny=("action", lambda x: (x == "Deny").sum()),
    ).reset_index()
    agg["deny_rate"] = agg["deny"] / agg["total_flux"]
    agg["statut"] = agg["deny_rate"].apply(
        lambda r: "Majoritairement Deny" if r >= 0.5 else "Majoritairement Allow"
    )

    fig = px.scatter(
        agg,
        x="distinct_dst",
        y="total_flux",
        size="total_flux",
        color="statut",
        color_discrete_map={
            "Majoritairement Allow": "#34d399",
            "Majoritairement Deny": "#f87171",
        },
        custom_data=["ipsrc"],
        hover_name="ipsrc",
        hover_data={
            "total_flux": True,
            "distinct_dst": True,
            "permit": True,
            "deny": True,
            "deny_rate": ":.2f",
            "statut": False,
        },
        title="IP Sources — Flux vs Destinations uniques",
        labels={
            "total_flux": "Nombre total de flux",
            "distinct_dst": "Destinations uniques",
            "statut": "Statut",
            "permit": "Autorisés",
            "deny": "Rejetés",
            "deny_rate": "Taux de refus",
        },
        size_max=40,
    )
    fig.update_layout(
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        title=dict(
            font=dict(color="#e2e8f0", size=16),
            subtitle=dict(
                text="Cliquez sur un point pour afficher le détail de l'IP",
                font=dict(color="#22d3ee", size=12),
            ),
        ),
        legend=dict(
            title=dict(text="Statut", font=dict(color="#94a3b8")),
            font=dict(color="#94a3b8"),
        ),
        xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", tickfont=dict(color="#94a3b8")),
    )
    return fig


def traffic_timeline_with_ip(df_raw: DataFrame, selected_ip: str) -> go.Figure:
    """Area chart of global network activity over time with a vertical marker
    at the first appearance of selected_ip.

    - Area fill: total flows per minute across all IPs
    - Green dashed vline: first seen timestamp of selected_ip
    """
    # Bucket by minute
    bucketed = df_raw.copy()
    bucketed["minute"] = bucketed["date"].dt.floor("min")
    traffic = bucketed.groupby("minute").size().reset_index(name="flux")

    # First seen for the selected IP
    ip_mask = df_raw["ipsrc"] == selected_ip
    first_seen = df_raw.loc[ip_mask, "date"].min() if ip_mask.any() else None
    # Plotly needs numeric ms-since-epoch for datetime axes (string ISO fails in add_vline)
    first_seen_ms = first_seen.floor("min").timestamp() * 1000 if first_seen is not None else None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=traffic["minute"],
        y=traffic["flux"],
        mode="lines",
        fill="tozeroy",
        name="Flux total",
        line=dict(color="#38bdf8", width=1.5),
        fillcolor="rgba(56, 189, 248, 0.12)",
        hovertemplate="<b>%{x|%H:%M}</b><br>%{y:,} flux<extra></extra>",
    ))

    if first_seen_ms is not None:
        fig.add_vline(
            x=first_seen_ms,
            line=dict(color="#34d399", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=first_seen_ms,
            y=0.97,
            xref="x",
            yref="paper",
            text=f"  {selected_ip}",
            font=dict(color="#34d399", size=11),

            bordercolor="#34d399",
            borderwidth=1,
            showarrow=False,
            xanchor="left",
            yanchor="top",
        )

    fig.update_layout(
        title=dict(
            text="Activité réseau globale dans le temps",
            font=dict(color="#e2e8f0", size=16),
            subtitle=dict(
                text="La ligne verte marque la première apparition de l'IP sélectionnée",
                font=dict(color="#34d399", size=12),
            ),
        ),
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        xaxis=dict(
            title="Temps",
            gridcolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title="Flux / minute",
            gridcolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=70, b=40, l=60, r=20),
    )
    return fig


def deny_permit_timeline(df_raw: DataFrame) -> go.Figure:
    """Line chart: Deny (red) vs Permit (green) flows per minute over time,
    with a rolling sum of Deny (orange dashed) to highlight attack peaks.
    """
    bucketed = df_raw.copy()
    bucketed["minute"] = bucketed["date"].dt.floor("min")

    grouped = (
        bucketed.groupby(["minute", "action"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    grouped.columns.name = None

    for col in ["Permit", "Deny"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped = grouped.sort_values("minute").reset_index(drop=True)

    grouped["total"] = grouped["Permit"] + grouped["Deny"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grouped["minute"],
        y=grouped["Permit"],
        name="Permit",
        mode="lines",
        line=dict(color="#34d399", width=2),
        hovertemplate="<b>%{x|%H:%M}</b><br>Permit : %{y:,}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=grouped["minute"],
        y=grouped["Deny"],
        name="Deny",
        mode="lines",
        line=dict(color="#f87171", width=2),
        hovertemplate="<b>%{x|%H:%M}</b><br>Deny : %{y:,}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=grouped["minute"],
        y=grouped["total"],
        name="Total (Permit + Deny)",
        mode="lines",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
        hovertemplate="<b>%{x|%H:%M}</b><br>Total : %{y:,}<extra></extra>",
    ))

    fig.update_layout(
        title="Évolution temporelle des flux Deny / Permit",
        xaxis=dict(title="Temps"),
        yaxis=dict(title="Flux / minute"),
        hovermode="x unified",
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig


def ip_rank_scatter(ip_agg: DataFrame, selected_ip: str) -> go.Figure:
    """Ranked scatter: X = IP rank (1→N sorted by total_flux desc), Y = total_flux.

    - Blue circle-open markers: allow-majority IPs (deny_rate < 0.5)
    - Red cross markers: deny-majority IPs (deny_rate >= 0.5)
    - Green vertical line at selected IP's rank
    - Annotation box showing details of the selected IP
    """
    df = ip_agg.reset_index(drop=True).copy()
    df["rank"] = range(1, len(df) + 1)
    df["deny_rate"] = df["deny"] / df["total_flux"]

    sel_mask = df["ipsrc"] == selected_ip
    sel_row = df[sel_mask].iloc[0] if sel_mask.any() else df.iloc[0]
    sel_rank = int(sel_row["rank"])

    deny_df = df[df["deny_rate"] >= 0.5]
    allow_df = df[df["deny_rate"] < 0.5]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=allow_df["rank"],
        y=allow_df["total_flux"],
        mode="markers",
        name="Majoritairement Allow",
        marker=dict(symbol="circle-open", color="#1c824a", size=7, line=dict(width=1.5)),
        customdata=allow_df[["ipsrc", "distinct_dst", "deny", "permit"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Flux totaux : %{y:,}<br>"
            "Destinations : %{customdata[1]}<br>"
            "Permit : %{customdata[3]} · Deny : %{customdata[2]}"
            "<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=deny_df["rank"],
        y=deny_df["total_flux"],
        mode="markers",
        name="Majoritairement Deny",
        marker=dict(symbol="cross", color="#f00e0e", size=7, line=dict(width=2)),
        customdata=deny_df[["ipsrc", "distinct_dst", "deny", "permit"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Flux totaux : %{y:,}<br>"
            "Destinations : %{customdata[1]}<br>"
            "Permit : %{customdata[3]} · Deny : %{customdata[2]}"
            "<extra></extra>"
        ),
    ))

    # Green vline at selected rank (integer x — no ISO string bug)
    fig.add_vline(x=sel_rank, line=dict(color="#34d399", width=2))

    fig.add_annotation(
        x=0.98, y=0.95,
        xref="paper", yref="paper",
        text=(
            f"<b>Nombre IP destination contactées : {int(sel_row['distinct_dst'])}</b><br>"
            f"Nombre de deny : {int(sel_row['deny'])}<br>"
            f"Adresse IP source : {selected_ip}"
        ),
        showarrow=False,
        align="left",
        bgcolor="#818cf8",
        bordercolor="#ffffff",
        borderwidth=1,
        font=dict(color="#FFFFFF", size=14),
        xanchor="right",
        yanchor="top",
    )

    fig.update_layout(
        font=dict(color="#000000"),
        title=dict(
            text="Classement des IP sources par volume de flux",
            font=dict(color="#000000", size=18),
        ),
        xaxis=dict(
            title="Rang (IPs triées par volume décroissant)",
            gridcolor="#000000",
            linecolor="#1e293b",
            tickfont=dict(color="#000000"),
        ),
        yaxis=dict(
            title="Nombre total de flux",
            gridcolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(color="#000000"),
        ),
        legend=dict(font=dict(color="#94a3b8")),
        hovermode="closest",
        height=450,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def top_dst_ports_bar(df_raw: DataFrame, top_n: int = 12) -> go.Figure:
    """Bar chart of the top N most targeted destination ports from raw logs.
    """
    _PALETTE = [
        "#22d3ee", "#34d399", "#f87171", "#818cf8", "#fbbf24",
        "#a78bfa", "#fb923c", "#38bdf8", "#4ade80", "#f472b6",
        "#facc15", "#e879f9",
    ]
    top = df_raw["portdst"].value_counts().head(top_n).reset_index()
    top.columns = ["port", "count"]
    top["port"] = top["port"].astype(str)

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top))]


    fig = go.Figure(go.Bar(
        x=top["port"],
        y=top["count"],
        marker_color=colors,
        hovertemplate="<b>Port %{x}</b><br>Occurrences : %{y:,}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} des ports destination les plus sollicités",
            font=dict(size=15, color="#000000"),
        ),
        xaxis=dict(
            title="Port destination",
            type="category",
            categoryorder="array",
            categoryarray=top["port"].tolist(),
            tickfont=dict( size=11),
            gridcolor="#acb5c5",
            linecolor="#acb5c5",
        ),
        yaxis=dict(
            title="Occurrences",
            tickfont=dict(size=11),
            gridcolor="#acb5c5",
            linecolor="#acb5c5",
        ),
        hoverlabel=dict(
            bgcolor="#1a2234",
            bordercolor="#334155",
            font=dict(size=12),
        ),
        bargap=0.2,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig

# =============================================================================
# TRAFFIC OVERVIEW (VISUALISATIONS)
# =============================================================================


def access_distribution(df: DataFrame) -> go.Figure:
    """Histogram of total connections per source IP (log scale).
    Helps identify high-volume sources that may be scanning or flooding.
    """
    fig = px.histogram(
        df,
        x="access_nbr",
        nbins=80,
        title="Distribution du nombre de connexions par IP source",
        labels={"access_nbr": "Nombre de connexions"},
        log_y=True,
    )
    fig.update_traces(marker_color="#4C9BE8")
    return fig


def deny_rate_distribution(df: DataFrame) -> go.Figure:
    """Histogram of deny rate across all source IPs.
    IPs clustered near 1.0 are almost entirely blocked — strong attack signal.
    """
    fig = px.histogram(
        df,
        x="deny_rate",
        nbins=50,
        title="Distribution du taux de refus (deny rate) par IP",
        labels={"deny_rate": "Taux de refus"},
    )
    fig.update_traces(marker_color="#E8684C")
    return fig


# =============================================================================
# SCANNING BEHAVIOUR
# =============================================================================


def horizontal_vs_vertical_scan(df: DataFrame) -> go.Figure:
    """Scatter plot of unique_dst_ratio vs unique_port_ratio.
    - Top-left  → vertical scan (many ports, few hosts)
    - Top-right → full scan (many ports AND many hosts)
    - Bottom-right → horizontal scan (many hosts, few ports)
    Color encodes deny_rate to surface blocked scanners.
    """
    fig = px.scatter(
        df.reset_index(),
        x="unique_dst_ratio",
        y="unique_port_ratio",
        color="deny_rate",
        color_continuous_scale="Reds",
        hover_name="ipsrc",
        hover_data={"access_nbr": True, "deny_nbr": True},
        title="Scanning behavior — ratio destinations vs ratio ports",
        labels={
            "unique_dst_ratio": "Ratio destinations uniques",
            "unique_port_ratio": "Ratio ports uniques",
            "deny_rate": "Taux de refus",
        },
    )
    return fig


def requests_per_second_top(df: DataFrame, top_n: int = 30) -> go.Figure:
    """Bar chart of the top N IPs by requests/second.
    High rps with high deny_rate → likely DoS or brute-force.
    """
    top = df.nlargest(top_n, "requests_per_second").reset_index()
    fig = px.bar(
        top,
        x="ipsrc",
        y="requests_per_second",
        color="deny_rate",
        color_continuous_scale="Reds",
        title=f"Top {top_n} IP par vélocité (requêtes/seconde)",
        labels={
            "ipsrc": "IP source",
            "requests_per_second": "Requêtes / seconde",
            "deny_rate": "Taux de refus",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# =============================================================================
# DENY / PERMIT ANALYSIS
# =============================================================================


def deny_vs_permit_bubble(df: DataFrame) -> go.Figure:
    """Bubble chart: permit_nbr vs deny_nbr, size = access_nbr.
    Surfaces IPs with massive deny counts that might still have some permits
    (partial access attempts or mixed legitimate/malicious traffic).
    """
    fig = px.scatter(
        df.reset_index(),
        x="permit_nbr",
        y="deny_nbr",
        size="access_nbr",
        color="deny_rate",
        color_continuous_scale="RdYlGn_r",
        hover_name="ipsrc",
        title="Permit vs Deny — volume de trafic par IP",
        labels={
            "permit_nbr": "Connexions autorisées",
            "deny_nbr": "Connexions refusées",
            "deny_rate": "Taux de refus",
        },
        size_max=40,
    )
    return fig


def sensitive_ports_top(df: DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of IPs with the most hits on sensitive ports (SSH, RDP, SMB…).
    Direct indicator of targeted intrusion attempts.
    """
    top = df.nlargest(top_n, "sensitive_ports_nbr").reset_index()
    fig = px.bar(
        top,
        x="ipsrc",
        y="sensitive_ports_nbr",
        color="deny_rate",
        color_continuous_scale="Reds",
        title=f"Top {top_n} IP par accès aux ports sensibles (SSH, RDP, SMB…)",
        labels={
            "ipsrc": "IP source",
            "sensitive_ports_nbr": "Accès ports sensibles",
            "deny_rate": "Taux de refus",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# =============================================================================
# RULES
# =============================================================================


def top_triggered_rules(df: DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of the most frequently triggered firewall rules.
    Concentrations on a few rules reveal systematic attack patterns.
    """
    rule_counts = df["most_triggered_rule"].value_counts().head(top_n).reset_index()
    rule_counts.columns = ["regle", "count"]
    rule_counts["regle"] = rule_counts["regle"].astype(str)

    fig = px.bar(
        rule_counts,
        x="regle",
        y="count",
        title=f"Top {top_n} règles pare-feu les plus déclenchées",
        labels={"regle": "Règle", "count": "Nombre d'IP"},
    )
    fig.update_traces(marker_color="#9B59B6")
    return fig


def deny_rules_distribution(df: DataFrame) -> go.Figure:
    """Histogram of distinct deny rules hit per IP.
    IPs triggering many different deny rules are probing broadly.
    """
    fig = px.histogram(
        df,
        x="deny_rules_hit",
        nbins=30,
        title="Nombre de règles de refus distinctes déclenchées par IP",
        labels={"deny_rules_hit": "Règles de refus distinctes"},
    )
    fig.update_traces(marker_color="#E67E22")
    return fig


# =============================================================================
# CLUSTERING
# =============================================================================


def corr_circle(result: ClusteringResult) -> go.Figure:
    # DataFrame des corrélations variables-composantes
    df = result.corr_plot

    theta = np.linspace(0, 2 * np.pi, 200)

    fig = go.Figure()

    # Cercle unité
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="royalblue", dash="dash"),
            name="Cercle unité"
        )
    )

    # Axes
    fig.add_shape(type="line", x0=-1.1, x1=1.1, y0=0, y1=0,
                  line=dict(color="gray", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-1.1, y1=1.1,
                  line=dict(color="gray", width=1))

    # Vecteurs des variables
    fig.add_trace(
        go.Scatter(
            x=[0] * len(df),
            y=[0] * len(df),
            mode="markers",
            marker=dict(size=1),
            showlegend=False
        )
    )

    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["PC1"],
            y=row["PC2"],
            ax=0,
            ay=0,
            text=row["variable"],
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="crimson"
        )

    fig.update_layout(
        title="Cercle de corrélation (PC1 vs PC2)",
        xaxis=dict(range=[-1.1, 1.1], zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], zeroline=False,
                   scaleanchor="x", scaleratio=1),
        width=700,
        height=700,
        showlegend=False
    )
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def scatter_3d_clusters(result: ClusteringResult) -> go.Figure:
    """3D scatter plot of clustering or anomaly detection results.

    - mode=="cluster": discrete color per cluster label (label -1 shown as "Bruit")
    - mode=="anomaly": continuous color scale (RdBu) by anomaly score
    """
    df = result.projection_plot

    # Clustering coloration
    if result.mode == "cluster":
        fig = px.scatter_3d(
            df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="cluster_str",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "cluster_str": False,
            },
            title=f"Clustering 3D — {result.algorithm} ({result.reducer.upper()})",
            labels={
                "pc1": "Composante 1",
                "pc2": "Composante 2",
                "pc3": "Composante 3",
            },
        )

    # Continuous anomaly score coloration
    else:
        fig = px.scatter_3d(
            df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="anomaly_score",
            color_continuous_scale="RdBu_r",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "anomaly_score": ":.4f",
            },
            title=f"Détection d'anomalies 3D — {result.algorithm} ({result.reducer.upper()})",
            labels={
                "pc1": "Composante 1",
                "pc2": "Composante 2",
                "pc3": "Composante 3",
                "anomaly_score": "Score d'anomalie",
            },
        )

    fig.update_traces(marker_size=4)
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def scatter_2d_clusters(result: ClusteringResult) -> go.Figure:
    """2D scatter plot of clustering or anomaly detection results (pc1 vs pc2)."""
    df = result.projection_plot

    if result.mode == "cluster":
        fig = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="cluster_str",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "cluster_str": False,
            },
            title=f"Clustering 2D — {result.algorithm} ({result.reducer.upper()})",
            labels={"pc1": "Composante 1", "pc2": "Composante 2"},
        )
    else:
        fig = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="anomaly_score",
            color_continuous_scale="RdBu_r",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "anomaly_score": ":.4f",
            },
            title=f"Détection d'anomalies 2D — {result.algorithm} ({result.reducer.upper()})",
            labels={
                "pc1": "Composante 1",
                "pc2": "Composante 2",
                "anomaly_score": "Score d'anomalie",
            },
        )

    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def dendrogram(result: ClusteringResult):
    df = result.projection_plot
    fake_X = np.zeros((len(df), 1))

    fig = ff.create_dendrogram(
        fake_X,
        linkagefun=lambda _: result.linkage,
        orientation='left',
        labels=None
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# =============================================================================
# FIREWALL DASHBOARD PANELS
# =============================================================================


def top5_ip_sources_bar(df_raw: DataFrame) -> go.Figure:
    """Horizontal stacked bar chart (Autorisés / Rejetés) for the TOP 5 source IPs by total flux."""
    agg = (
        df_raw.groupby("ipsrc")
        .agg(
            permit=("action", lambda x: (x == "Permit").sum()),
            deny=("action", lambda x: (x == "Deny").sum()),
        )
        .reset_index()
    )
    agg["total"] = agg["permit"] + agg["deny"]
    top5 = agg.nlargest(5, "total").sort_values("total")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top5["ipsrc"], x=top5["permit"],
        name="Autorisés", orientation="h",
        marker_color="#34d399",
    ))
    fig.add_trace(go.Bar(
        y=top5["ipsrc"], x=top5["deny"],
        name="Rejetés", orientation="h",
        marker_color="#f87171",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(color="#334155"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="left", x=0, font=dict(color="#64748b"),
        ),
        margin=dict(t=10, b=60, l=130, r=20),
        height=300,
    )
    return fig


def top10_permitted_ports_bar(df_raw: DataFrame) -> go.Figure:
    """Vertical bar chart for the TOP 10 destination ports < 1024 with permitted access."""
    _PALETTE = [
        "#22d3ee", "#34d399", "#f87171", "#818cf8", "#fbbf24",
        "#a78bfa", "#fb923c", "#38bdf8", "#4ade80", "#f472b6",
    ]
    permitted = df_raw[(df_raw["action"] == "Permit") & (df_raw["portdst"] < 1024)]
    port_counts = permitted["portdst"].value_counts().head(10).reset_index()
    port_counts.columns = ["port", "count"]
    port_counts["port"] = port_counts["port"].astype(str)

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(port_counts))]

    fig = go.Figure(go.Bar(
        x=port_counts["port"],
        y=port_counts["count"],
        marker_color=colors,
        hovertemplate="<b>Port %{x}</b><br>Occurrences : %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(
            title="Port destination",
            type="category",
            categoryorder="array",
            categoryarray=port_counts["port"].tolist(),
            tickfont=dict(size=11),
            gridcolor="#acb5c5",
            linecolor="#acb5c5",
        ),
        yaxis=dict(
            title="Occurrences",
            tickfont=dict(size=11),
            gridcolor="#acb5c5",
            linecolor="#acb5c5",
        ),
        hoverlabel=dict(
            bgcolor="#1a2234",
            bordercolor="#334155",
            font=dict(size=12),
        ),
        bargap=0.2,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig

def line_cluster_inertia(inertia: list, total_inertia: int|None = None) -> go.Figure:
    fig = px.line(y=inertia, title="Inertie par nombre de cluster")
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if total_inertia:
        fig.update_layout(
            xaxis=dict(range=[1, 10], dtick=1)
        )

    return fig