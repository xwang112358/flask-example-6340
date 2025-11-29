# UMAP Visualization Implementation Summary

## What Was Implemented

An interactive UMAP visualization system for analyzing active molecules in RNA target screening results using ChemPlot.

## Key Features

✅ **Pre-computed Caching System**
- UMAP embeddings are computed once on server startup
- Cached to disk as JSON files in `umap_cache/` directory
- Instant loading (~100ms instead of 30+ seconds)

✅ **Interactive Plotly Visualizations**
- Hover tooltips showing molecule name and coordinates
- Zoom and pan capabilities
- Clean, professional styling

✅ **Error Handling**
- Invalid target validation
- Minimum molecule count checks (need ≥3 molecules)
- Missing dependency detection
- User-friendly error messages

✅ **Performance Optimized**
- First load: ~30 seconds per target (one-time computation)
- Subsequent loads: Instant (from cache)
- Background initialization doesn't block server startup

## Data Cached

| Target | Active Molecules | Cache File Size |
|--------|-----------------|----------------|
| TPP | 162 molecules | 16 KB |
| Glutamine_RS | 70 molecules | 6.5 KB |
| ZTP | 170 molecules | 16 KB |
| SAM_ll | 196 molecules | 18.7 KB |
| PreQ1 | 166 molecules | 15.9 KB |

## File Structure

```
flask-example-6340/
├── server.py                    # Flask server with caching logic
├── requirements.txt             # Dependencies (chemplot, rdkit, plotly)
├── templates/
│   └── analyze.html            # Updated with UMAP visualization
├── umap_cache/                 # Auto-generated cache directory
│   ├── TPP.json
│   ├── Glutamine_RS.json
│   ├── ZTP.json
│   ├── SAM_ll.json
│   └── PreQ1.json
└── .gitignore                  # Excludes cache from version control
```

## How It Works

### Server Startup
1. Server initializes and loads dataset
2. Background thread checks for cached UMAP files
3. If cache missing, computes UMAP for all targets (~2-3 minutes total)
4. Saves embeddings as JSON files

### User Request Flow
1. User selects RNA target and clicks "Analyze"
2. Analysis page loads with hit count and examples
3. JavaScript fetches UMAP data from `/umap_plot` endpoint
4. Server loads cached JSON file (instant)
5. Creates Plotly figure from cached coordinates
6. Returns JSON to browser
7. Interactive plot renders

### Cache Management

**View cached data:**
- Cache files located in `umap_cache/` directory
- Each file contains: `umap_x`, `umap_y`, `names`, `smiles`

**Regenerate cache for specific target:**
```
Visit: http://localhost:5000/regenerate_umap/<target>
Example: http://localhost:5000/regenerate_umap/TPP
```

**Clear all cache:**
```bash
# Delete cache directory
rm -rf umap_cache/
# Or on Windows:
rmdir /s umap_cache
```

Server will automatically regenerate cache on next startup.

## Technical Details

### Dependencies
- **chemplot**: UMAP generation and molecular descriptors
- **rdkit**: Chemical structure processing
- **plotly**: Interactive visualization
- **flask**: Web framework
- **pandas**: Data manipulation

### UMAP Parameters
- Algorithm: UMAP (Uniform Manifold Approximation and Projection)
- Similarity type: Structural (based on molecular descriptors)
- Default parameters: ChemPlot defaults (n_neighbors=15, min_dist=0.1)

### Visualization Styling
- Color: Blue (#2196F3)
- Marker size: 8px
- Opacity: 0.7
- Border: White, 1px
- Template: plotly_white (clean, minimal)

## Usage

1. **Start server:**
   ```bash
   python server.py
   ```

2. **Access web app:**
   - Open http://localhost:5000

3. **Analyze target:**
   - Select RNA target from dropdown
   - Click "Analyze"
   - Wait for UMAP to load (instant if cached)
   - Interact with visualization:
     - Hover over points to see molecule details
     - Zoom/pan to explore clusters
     - Identify structural similarity patterns

## Future Enhancements (Not Implemented)

Potential improvements for later:
- Color points by molecular properties (MW, LogP)
- Show 2D molecular structures on hover
- Compare multiple targets side-by-side
- Filter by similarity clusters
- Download selected molecules as CSV
- Export plot as PNG/SVG
- Add inactive molecules for comparison

## Troubleshooting

**UMAP not loading:**
- Check server console for errors
- Verify cache files exist in `umap_cache/`
- Try regenerating cache for that target

**No data points in plot:**
- This issue has been fixed
- Verify cache contains `umap_x` and `umap_y` arrays
- Check browser console for JavaScript errors

**Slow initial load:**
- Normal on first run (computing UMAPs)
- Cache initialization runs in background
- Subsequent loads will be instant

**Cache not initializing:**
- Check server console output
- Ensure write permissions for `umap_cache/` directory
- Verify all dependencies installed correctly

