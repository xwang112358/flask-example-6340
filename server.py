from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import json
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, QED, Descriptors
import base64
from io import BytesIO
import numpy as np
import umap
import plotly.graph_objects as go
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the dataset once when the server starts
df = pd.read_csv("robin_clean.csv")

# Cache directory for UMAP embeddings
CACHE_DIR = Path("umap_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache file for molecular properties (QED and SA scores)
PROPERTIES_CACHE_FILE = CACHE_DIR / "molecular_properties.json"

def smiles_to_image_base64(smiles, size=(200, 200)):
    """Convert SMILES to base64-encoded PNG image for hover display."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating image for SMILES {smiles}: {e}")
        return None

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    """Convert SMILES to Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Use MorganGenerator (new API) instead of deprecated GetMorganFingerprintAsBitVect
        generator = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = generator.GetFingerprint(mol)
        return np.array(fp)
    except Exception as e:
        print(f"Error generating fingerprint for SMILES {smiles}: {e}")
        return None

def compute_and_cache_umap(target):
    """Compute UMAP embeddings for a target and cache to disk."""
    print(f"Computing UMAP for {target}...")
    
    # Get active molecules
    active_df = df[df[target] == 1]
    
    if len(active_df) < 3:
        return None
    
    smiles_list = active_df["Smile"].tolist()
    names_list = active_df["Name"].tolist()
    
    try:
        # Generate Morgan fingerprints for all molecules
        print(f"Generating molecular fingerprints for {target}...")
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            fp = smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        if len(fingerprints) < 3:
            print(f"Not enough valid molecules for {target}")
            return None
        
        # Filter to only valid molecules
        smiles_list = [smiles_list[i] for i in valid_indices]
        names_list = [names_list[i] for i in valid_indices]
        
        # Convert to numpy array
        fingerprints = np.array(fingerprints)
        
        # Compute UMAP embedding
        print(f"Computing UMAP embedding for {target}...")
        n_neighbors = min(15, len(fingerprints) - 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            metric='jaccard'
        )
        embedding = reducer.fit_transform(fingerprints)
        
        # Generate base64 images for each molecule
        print(f"Generating molecule images for {target}...")
        images_list = [smiles_to_image_base64(smiles) for smiles in smiles_list]
        
        # Extract coordinates as lists
        cache_data = {
            'umap_x': embedding[:, 0].tolist(),
            'umap_y': embedding[:, 1].tolist(),
            'names': names_list,
            'smiles': smiles_list,
            'images': images_list
        }
        
        # Save to cache
        cache_file = CACHE_DIR / f"{target}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"✓ Cached UMAP for {target} ({len(names_list)} molecules)")
        return cache_data
        
    except Exception as e:
        print(f"✗ Error computing UMAP for {target}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_cached_umap(target):
    """Load UMAP embeddings from cache."""
    cache_file = CACHE_DIR / f"{target}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def compute_molecular_properties():
    """Compute QED and SA scores for all molecules and cache them."""
    print("Computing molecular properties (QED and SA scores)...")
    
    all_smiles = df["Smile"].tolist()
    properties = {
        'qed': [],
        'sa': [],
        'smiles': []
    }
    
    for idx, smiles in enumerate(all_smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Calculate QED
                qed_value = QED.qed(mol)
                
                # Calculate SA Score approximation
                num_rings = Descriptors.RingCount(mol)
                num_heteroatoms = Descriptors.NumHeteroatoms(mol)
                num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                mol_weight = Descriptors.MolWt(mol)
                
                sa_score = 1 + (num_rings * 0.5) + (num_heteroatoms * 0.3) + (num_rotatable_bonds * 0.2) + (mol_weight / 100)
                sa_score = min(10, max(1, sa_score))
                
                properties['qed'].append(qed_value)
                properties['sa'].append(sa_score)
                properties['smiles'].append(smiles)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(all_smiles)} molecules...")
    
    # Save to cache
    with open(PROPERTIES_CACHE_FILE, 'w') as f:
        json.dump(properties, f)
    
    print(f"✓ Cached molecular properties for {len(properties['qed'])} molecules")
    return properties

def load_cached_properties():
    """Load cached molecular properties."""
    if PROPERTIES_CACHE_FILE.exists():
        with open(PROPERTIES_CACHE_FILE, 'r') as f:
            return json.load(f)
    return None

# Pre-compute UMAP embeddings for all targets on startup
def initialize_umap_cache():
    """Pre-compute and cache UMAP embeddings for all targets."""
    valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
    print("\n" + "="*50)
    print("Initializing UMAP cache...")
    print("="*50)
    
    for target in valid_targets:
        cache_data = load_cached_umap(target)
        if cache_data is None:
            compute_and_cache_umap(target)
        else:
            print(f"✓ Using cached UMAP for {target} ({len(cache_data['names'])} molecules)")
    
    print("="*50)
    print("UMAP cache initialization complete!")
    
    # Also initialize molecular properties cache
    print("="*50)
    print("Initializing molecular properties cache...")
    print("="*50)
    
    properties = load_cached_properties()
    if properties is None:
        compute_molecular_properties()
    else:
        print(f"✓ Using cached molecular properties ({len(properties['qed'])} molecules)")
    
    print("="*50)
    print("Cache initialization complete!\n")

# Initialize cache on startup
# Run this in a separate thread to not block server startup
def init_cache_async():
    import threading
    def run_init():
        import time
        time.sleep(1)  # Give server time to start
        initialize_umap_cache()
    thread = threading.Thread(target=run_init, daemon=True)
    thread.start()

# Only initialize on the main process (not on reloader)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    init_cache_async()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    target = request.form["target"]
    
    # PDB ID mapping
    pdb_ids = {
        "TPP": "2GDI",
        "Glutamine_RS": "6QN3",
        "ZTP": "5BTP",
        "SAM_ll": "2QWY",
        "PreQ1": "3FU2"
    }
    
    # Check if target is valid
    valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
    num_hits = 0
    if target not in valid_targets:
        result = f"Invalid target. Please choose from: {', '.join(valid_targets)}"
    else:
        # Count active molecules (where target column equals 1)
        num_hits = (df[target] == 1).sum()
        pdb_id = pdb_ids[target]
        result = ""
        
        # Fetch PDB title from RCSB API
        pdb_title = "Title not available"
        try:
            api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                pdb_title = data.get("struct", {}).get("title", "Title not available")
        except Exception as e:
            pdb_title = f"Error fetching title: {str(e)}"
        
        # Get all active molecules
        active_molecules = df[df[target] == 1]["Smile"].tolist()
        examples = "<br>".join(active_molecules) if active_molecules else "No active molecules found"
    
    return render_template("analyze.html", analysis=result, target=target, examples=examples, 
                         pdb_id=pdb_id, pdb_title=pdb_title, num_hits=num_hits)


@app.route("/regenerate_umap/<target>")
def regenerate_umap(target):
    """Manually regenerate UMAP cache for a specific target."""
    valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
    if target not in valid_targets:
        return jsonify({"error": f"Invalid target: {target}"}), 400
    
    cache_data = compute_and_cache_umap(target)
    if cache_data:
        return jsonify({"success": f"UMAP regenerated for {target}", "num_molecules": len(cache_data['names'])})
    else:
        return jsonify({"error": f"Failed to generate UMAP for {target}"}), 500


@app.route("/umap_plot", methods=["POST"])
def umap_plot():
    try:
        target = request.form["target"]
        
        # Validate target
        valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
        if target not in valid_targets:
            return jsonify({"error": f"Invalid target: {target}"}), 400
        
        # Load cached UMAP data
        cache_data = load_cached_umap(target)
        
        if cache_data is None:
            return jsonify({"error": f"No UMAP data available for {target}. Not enough active molecules (need at least 3)."}), 400
        
        # Load cached molecular properties
        properties = load_cached_properties()
        if properties is None:
            return jsonify({"error": "Molecular properties not cached. Please restart the server."}), 500
        
        # Create a mapping from SMILES to properties
        smiles_to_props = {
            smiles: {'qed': qed, 'sa': sa}
            for smiles, qed, sa in zip(properties['smiles'], properties['qed'], properties['sa'])
        }
        
        # Add QED and SA scores to cache_data
        qed_values = []
        sa_values = []
        for smiles in cache_data['smiles']:
            if smiles in smiles_to_props:
                qed_values.append(smiles_to_props[smiles]['qed'])
                sa_values.append(smiles_to_props[smiles]['sa'])
            else:
                qed_values.append(None)
                sa_values.append(None)
        
        cache_data['qed'] = qed_values
        cache_data['sa'] = sa_values
        
        # Create hover text with molecule information
        hover_texts = []
        for i in range(len(cache_data['names'])):
            hover_text = (
                f"<b>{cache_data['names'][i]}</b><br>"
                f"UMAP-1: {cache_data['umap_x'][i]:.2f}<br>"
                f"UMAP-2: {cache_data['umap_y'][i]:.2f}<br>"
                f"SMILES: {cache_data['smiles'][i]}<br>"
                f"<extra></extra>"  # Removes the secondary box in plotly
            )
            hover_texts.append(hover_text)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add scatter trace with custom data for images
        fig.add_trace(go.Scatter(
            x=cache_data['umap_x'],
            y=cache_data['umap_y'],
            mode='markers',
            marker=dict(
                size=10,
                color='#2196F3',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=hover_texts,
            hovertemplate='%{text}',
            customdata=[[img, name, smiles] for img, name, smiles in 
                       zip(cache_data['images'], cache_data['names'], cache_data['smiles'])],
            name=''
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"UMAP Visualization of Active Molecules for {target} ({len(cache_data['names'])} molecules)",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis=dict(
                title="UMAP-1",
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False
            ),
            yaxis=dict(
                title="UMAP-2",
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False
            ),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            width=900,
            height=600,
            hovermode='closest',
            showlegend=False
        )
        
        # Return plot data as JSON instead of HTML string
        # This avoids script execution issues when injecting HTML
        return jsonify({
            "plot_data": json.loads(fig.to_json()),
            "images": cache_data['images'],
            "names": cache_data['names'],
            "smiles": cache_data['smiles'],
            "qed": cache_data['qed'],
            "sa": cache_data['sa']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating UMAP plot: {str(e)}"}), 500


@app.route("/kmeans_cluster", methods=["POST"])
def kmeans_cluster():
    """Perform k-means clustering on ECFP fingerprints of active molecules."""
    try:
        target = request.form["target"]
        
        # Validate target
        valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
        if target not in valid_targets:
            return jsonify({"error": f"Invalid target: {target}"}), 400
        
        # Parse and validate n_clusters
        try:
            n_clusters = int(request.form["n_clusters"])
        except (ValueError, KeyError):
            return jsonify({"error": "Number of clusters must be a valid integer"}), 400
        
        # Validate n_clusters range
        if n_clusters < 2:
            return jsonify({"error": "Number of clusters must be at least 2"}), 400
        
        if n_clusters > 50:
            return jsonify({"error": "Number of clusters cannot exceed 50"}), 400
        
        # Load cached UMAP data to get the molecule order
        cache_data = load_cached_umap(target)
        
        if cache_data is None:
            return jsonify({"error": f"No UMAP data available for {target}"}), 400
        
        # Get the SMILES in the same order as the cached UMAP data
        smiles_list = cache_data['smiles']
        
        # Check if we have enough molecules for clustering
        if len(smiles_list) < n_clusters:
            return jsonify({"error": f"Not enough molecules ({len(smiles_list)}) for {n_clusters} clusters"}), 400
        
        # Generate ECFP fingerprints for all molecules
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            fp = smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        if len(fingerprints) < n_clusters:
            return jsonify({"error": f"Not enough valid fingerprints ({len(fingerprints)}) for {n_clusters} clusters"}), 400
        
        # Convert to numpy array
        fingerprints = np.array(fingerprints)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(fingerprints)
        
        # Convert cluster labels to list (they should match the order of UMAP data)
        cluster_labels_list = cluster_labels.tolist()
        
        return jsonify({
            "cluster_labels": cluster_labels_list,
            "n_clusters": n_clusters,
            "n_molecules": len(cluster_labels_list)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error performing k-means clustering: {str(e)}"}), 500


@app.route("/density_plots", methods=["POST"])
def density_plots():
    """Get QED and SA Score distributions for all molecules and positive hits from cache."""
    try:
        target = request.form["target"]
        
        # Validate target
        valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
        if target not in valid_targets:
            return jsonify({"error": f"Invalid target: {target}"}), 400
        
        # Load cached properties
        properties = load_cached_properties()
        if properties is None:
            return jsonify({"error": "Molecular properties not cached. Please restart the server."}), 500
        
        # Create a mapping from SMILES to properties for quick lookup
        smiles_to_props = {
            smiles: {'qed': qed, 'sa': sa}
            for smiles, qed, sa in zip(properties['smiles'], properties['qed'], properties['sa'])
        }
        
        # Get positive hits for this target
        positive_hits_df = df[df[target] == 1]
        positive_hits_smiles = positive_hits_df["Smile"].tolist()
        
        # Extract properties for positive hits
        qed_hits = []
        sa_hits = []
        
        for smiles in positive_hits_smiles:
            if smiles in smiles_to_props:
                qed_hits.append(smiles_to_props[smiles]['qed'])
                sa_hits.append(smiles_to_props[smiles]['sa'])
        
        return jsonify({
            "qed_all": properties['qed'],
            "qed_hits": qed_hits,
            "sa_all": properties['sa'],
            "sa_hits": sa_hits,
            "n_all": len(properties['qed']),
            "n_hits": len(qed_hits)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error loading density plots: {str(e)}"}), 500


if __name__ == "__main__":
    # Enable debug mode and the reloader so changes are picked up automatically
    # During development you can run with: python server.py
    # If you prefer the flask CLI, set FLASK_APP and FLASK_DEBUG (see README/run notes below)
    app.run(debug=True, use_reloader=True)

    # Optional: to auto-refresh the browser (no manual refresh) install `livereload`:
    #   pip install livereload
    # Then replace the app.run(...) above with the lines below (or uncomment and run):
    # from livereload import Server
    # server = Server(app.wsgi_app)
    # server.watch('templates/')
    # server.watch('static/')
    # server.watch('*.py')
    # server.serve(port=5000, host='127.0.0.1')
