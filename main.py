from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tempfile
import os
import traceback
import json

import MDAnalysis as mda

# ---------------- CONFIG ----------------
MAX_FILE_SIZE = 500 * 1024  # 500 KB
app = FastAPI(title="Nucleic Acid 3D Viewer")
templates = Jinja2Templates(directory="templates")


# ---------------- MIDDLEWARE ----------------
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        return HTMLResponse(status_code=413, content="File too large")
    return await call_next(request)


# ---------------- PDB PARSER ----------------
def parse_pdb(pdb_path: str):
    print(f"[DEBUG] Loading PDB: {pdb_path}")
    u = mda.Universe(pdb_path)
    print(f"[DEBUG] Total atoms: {len(u.atoms)}")

    atoms_data = []
    residues_data = []
    seen_residues = {}

    for atom in u.atoms:
        try:
            resname  = str(atom.resname).strip()
            resid    = int(atom.resid)
            name     = str(atom.name).strip()
            element  = str(atom.element).strip() if atom.element else name[0]
            x, y, z  = float(atom.position[0]), float(atom.position[1]), float(atom.position[2])

            try:
                chainid = str(atom.segid).strip() or "A"
            except Exception:
                chainid = "A"

            atoms_data.append({
                "index":   int(atom.index),
                "name":    name,
                "element": element,
                "resname": resname,
                "resid":   resid,
                "chain":   chainid,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
            })

            res_key = f"{chainid}:{resid}:{resname}"
            if res_key not in seen_residues:
                seen_residues[res_key] = True
                residues_data.append({
                    "resname": resname,
                    "resid":   resid,
                    "chain":   chainid,
                })
        except Exception as e:
            print(f"[WARN] Skipping atom: {e}")
            continue

    # Read raw PDB text for 3Dmol
    with open(pdb_path, "r") as f:
        pdb_text = f.read()

    # Detect molecule type
    na_resnames = {"DA","DT","DG","DC","DU","A","U","G","C","T","ADE","URA","GUA","CYT","THY","RA","RU","RG","RC"}
    mol_type = "nucleic" if any(r["resname"] in na_resnames for r in residues_data) else "protein"

    stats = {
        "atoms":    len(atoms_data),
        "residues": len(residues_data),
        "chains":   len(set(r["chain"] for r in residues_data)),
        "mol_type": mol_type,
    }

    print(f"[DEBUG] Parsed: {stats}")
    return pdb_text, atoms_data, residues_data, stats


# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    tmp_path = None
    try:
        print(f"[DEBUG] Upload: {file.filename}")
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "File exceeds 500 KB limit."
            })

        if not file.filename.lower().endswith(".pdb"):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Only .pdb files are accepted."
            })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        pdb_text, atoms_data, residues_data, stats = parse_pdb(tmp_path)

        return templates.TemplateResponse("index.html", {
            "request":       request,
            "pdb_text":      json.dumps(pdb_text),
            "atoms_json":    json.dumps(atoms_data),
            "residues_json": json.dumps(residues_data),
            "stats":         stats,
            "filename":      file.filename,
        })

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Processing failed: {str(e)}",
        })

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
