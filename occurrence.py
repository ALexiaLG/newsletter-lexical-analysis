#!/usr/bin/env python3
"""
occurrence.py

Very stupid lexical analysis for titles.csv.

Features:
- Per-year word occurrence
- Animated top-N words (Plotly)
- Dynamic lexical adjacency graph (Plotly)

Usage:
  python occurrence.py titles.csv --animate --graph
"""

from __future__ import annotations

import argparse
import re
import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Iterable

import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Tokenization
# -----------------------------

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")

GENERIC_STOP = {
    "a","an","the","and","or","of","to","in","for","with","on","by","from","at","as","is","are","be",
    "this","that","these","those","using","via","based","towards",
}

DOMAIN_STOP = {
    "soft","robot","robots","robotic","robotics",
    "inspired","design","designs","system","systems",
    "development","study","studies","approach","approaches",
    "method","methods",
}


def normalize_token(t: str) -> str:
    for suf in ("es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[:-len(suf)]
    for suf in ("ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[:-len(suf)]
    return t


def tokenize(text: str) -> list[str]:
    text = (text or "").replace("\u00A0", " ")
    toks = TOKEN_RE.findall(text)
    toks = [normalize_token(t.lower()) for t in toks]
    return [t for t in toks if t not in GENERIC_STOP and t not in DOMAIN_STOP]


# -----------------------------
# Occurrence analysis
# -----------------------------

def count_words(titles: Iterable[str]) -> Counter:
    c = Counter()
    for t in titles:
        c.update(tokenize(t))
    return c


def yearly_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("year"):
        c = count_words(g["title"])
        for w, n in c.items():
            rows.append((int(year), w, n))
    return pd.DataFrame(rows, columns=["year", "word", "count"])


def top_n_over_time(yearly_df, n=10, vocab_size=30):
    vocab = (
        yearly_df.groupby("word")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(vocab_size)
        .index
    )
    df = yearly_df[yearly_df["word"].isin(vocab)]
    return (
        df.sort_values(["year","count"], ascending=[True,False])
          .groupby("year")
          .head(n)
          .reset_index(drop=True)
    )


# -----------------------------
# Lexical adjacency
# -----------------------------

def build_adjacency(df: pd.DataFrame):
    edge_counts = defaultdict(lambda: defaultdict(int))
    node_counts = defaultdict(lambda: defaultdict(int))

    for _, r in df.iterrows():
        year = int(r["year"])
        toks = tokenize(r["title"])

        for t in toks:
            node_counts[t][year] += 1

        for a, b in pairwise(toks):
            if a == b:
                continue
            e = tuple(sorted((a, b)))
            edge_counts[e][year] += 1

    return edge_counts, node_counts


def build_graph(edge_counts, min_total=5):
    G = nx.Graph()
    for (a, b), years in edge_counts.items():
        if sum(years.values()) >= min_total:
            G.add_edge(a, b)
    return G


# -----------------------------
# Plotly: top-N animation
# -----------------------------

def animate_top_words(anim_df, out_html):
    fig = px.bar(
        anim_df,
        x="count",
        y="word",
        animation_frame="year",
        orientation="h",
        range_x=[0, anim_df["count"].max() * 1.1],
    )

    fig.update_layout(
        title="Top words in Soft Robotics titles (per year)",
        yaxis=dict(categoryorder="total ascending"),
        showlegend=False,
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}")


# -----------------------------
# Plotly: dynamic lexical graph
# -----------------------------

def plotly_lexical_graph(G, edge_counts, node_counts, years, out_html):

    pos = nx.spring_layout(G, seed=0)

    frames = []

    for year in years:
        edge_x, edge_y = [], []
        node_x, node_y, node_size, node_text = [], [], [], []

        for (u, v), d in edge_counts.items():
            c = d.get(year, 0)
            if c == 0 or not G.has_edge(u, v):
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        for n in G.nodes():
            c = node_counts[n].get(year, 0)
            if c == 0:
                continue
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            node_size.append(8 + 6 * math.sqrt(c))
            node_text.append(f"{n} ({c})")

        frames.append(
            go.Frame(
                name=str(year),
                data=[
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        mode="lines",
                        line=dict(width=1, color="rgba(0,0,0,0.3)"),
                        hoverinfo="none",
                    ),
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode="markers+text",
                        text=node_text,
                        textposition="top center",
                        marker=dict(size=node_size, color="steelblue"),
                    )
                ],
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="Lexical adjacency graph (per year)",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            sliders=[{
                "steps": [
                    {"method": "animate", "args": [[str(y)]], "label": str(y)}
                    for y in years
                ]
            }]
        )
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["year"])

    yearly_df = yearly_occurrence(df)
    years = sorted(yearly_df["year"].unique())

    if args.animate:
        anim_df = top_n_over_time(yearly_df)
        animate_top_words(anim_df, "top_words_animation.html")

    if args.graph:
        edge_counts, node_counts = build_adjacency(df)
        G = build_graph(edge_counts, min_total=5)
        plotly_lexical_graph(
            G, edge_counts, node_counts, years,
            out_html="lexical_graph_dynamic.html"
        )


if __name__ == "__main__":
    main()
