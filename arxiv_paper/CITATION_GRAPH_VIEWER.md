# arXiv Citation Graph Viewer

This repo now includes a Gradio app for the balanced arXiv citation graphs in:

```text
data/arxiv_citation_graph_balanced_1000
```

Run locally:

```bash
pip install -r requirements.txt
python app.py
```

Or run the module directly with an explicit data directory:

```bash
python -m arxiv_paper.citation_graph_viewer \
  --data-dir data/arxiv_citation_graph_balanced_1000 \
  --server-port 7860
```

For a Hugging Face Space, use the repository root as the Space root, or copy
`app.py`, `requirements.txt`, `arxiv_paper/citation_graph_viewer.py`, and the
`data/arxiv_citation_graph_balanced_1000` directory into the Space. If the data
is mounted elsewhere, set:

```bash
ARXIV_GRAPH_DATA_DIR=/path/to/arxiv_citation_graph_balanced_1000
```

The viewer is built around the citation graph semantics from the builders:

```text
referenced/cited paper -> later paper that cites it
```

The plot places papers on the x-axis by publication year, separates them by
primary category lanes, colors seed/reference roles, and lets you switch between
overview, seed/reference, most-cited, most-connected, and selected-paper
neighborhood views.
