from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import tempfile
import os
import json

import MDAnalysis as mda
import networkx as nx
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
MAX_FILE_SIZE = 100 * 1024  # 100 KB
MAX_ATOMS = 150

app = FastAPI(title="PDB Network Analyzer")
templates = Jinja2Templates(directory="templates")


# ---------------- MIDDLEWARE ----------------
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        return JSONResponse(
            {"error": "File too large. Maximum size is 100 KB."},
            status_code=413
        )
    return await call_next(request)


# ---------------- GRAPH BUILDER ----------------
def build_graph(pdb_path: str) -> nx.Graph:
    u = mda.Universe(pdb_path)

    # Prefer backbone CA atoms for clean, compact representation
    atoms = u.select_atoms("name CA")
    if len(atoms) == 0:
        atoms = u.atoms

    # Filter hydrogens and cap atom count
    atoms = [a for a in atoms if a.element != "H"][:MAX_ATOMS]

    if len(atoms) < 2:
        raise ValueError("Not enough atoms to build a graph (minimum: 2).")

    G = nx.Graph()

    for atom in atoms:
        G.add_node(
            atom.index,
            label=atom.name,
            resname=getattr(atom, "resname", "UNK"),
            resid=getattr(atom, "resid", 0),
        )

    # Distance-based edges (vectorised inner loop)
    for i, a1 in enumerate(atoms):
        for j, a2 in enumerate(atoms[i + 1:], start=i + 1):
            dist = float(((a1.position - a2.position) ** 2).sum() ** 0.5)
            if dist < 4.5:          # wider cutoff catches more real contacts
                G.add_edge(a1.index, a2.index, weight=round(dist, 2))

    return G


# ---------------- GRAPH STATS ----------------
def compute_stats(G: nx.Graph) -> dict:
    degrees = dict(G.degree())
    avg_deg = round(sum(degrees.values()) / max(len(G.nodes()), 1), 2)

    # Connected components
    components = list(nx.connected_components(G))

    # Clustering coefficient (fast approximation)
    clustering = round(nx.average_clustering(G), 4)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_degree": avg_deg,
        "components": len(components),
        "clustering": clustering,
    }


# ---------------- PLOT GRAPH ----------------
def plot_graph(G: nx.Graph) -> str:
    pos = nx.spring_layout(G, iterations=50, seed=42, k=0.8)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text = [], [], []
    degrees = dict(G.degree())
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        d = G.nodes[n]
        node_text.append(
            f"<b>{d.get('resname','?')}{d.get('resid','')}</b> · {d.get('label','')}<br>Degree: {degrees[n]}"
        )

    node_degrees = [degrees[n] for n in G.nodes()]
    max_deg = max(node_degrees) if node_degrees else 1

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(150,150,160,0.35)"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes — coloured by degree
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=[6 + 8 * (d / max_deg) for d in node_degrees],
            color=node_degrees,
            colorscale="Teal",
            showscale=True,
            colorbar=dict(
                title="Degree",
                thickness=10,
                len=0.5,
                tickfont=dict(size=10),
            ),
            line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
        ),
        text=node_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="SF Pro Display, -apple-system, Helvetica Neue, sans-serif"),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="SF Pro Display, -apple-system, Helvetica Neue, sans-serif",
        ),
    )

    return fig.to_html(full_html=False, config={"displayModeBar": False, "responsive": True})


# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    tmp_path = None
    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "File exceeds 100 KB limit. Please upload a smaller PDB file.",
            })

        if not file.filename.lower().endswith(".pdb"):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Only .pdb files are accepted.",
            })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        G = build_graph(tmp_path)
        stats = compute_stats(G)
        graph_html = plot_graph(G)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "graph": graph_html,
            "stats": stats,
            "filename": file.filename,
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
        })

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)