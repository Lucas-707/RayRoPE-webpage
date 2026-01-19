import re
import json
import os
from pathlib import Path

def extract_plotly_data(html_file):
    """Extract the data and layout from HTML file"""
    with open(html_file, 'r') as f:
        content = f.read()
    
    # Find Plotly.newPlot call
    match = re.search(r'Plotly\.newPlot\s*\(\s*"[^"]+",\s*(\[.*?\]),\s*(\{.*?\}),\s*\{[^}]*\}\s*\)', 
                      content, re.DOTALL)
    
    if match:
        data_str = match.group(1)
        layout_str = match.group(2)
        
        # Parse as JSON
        data = json.loads(data_str)
        layout = json.loads(layout_str)
        
        return {
            'data': data,
            'layout': layout
        }
    return None

# Process all HTML files
html_dir = Path('assets/htmls')
data_dir = Path('assets/stereo')
data_dir.mkdir(parents=True, exist_ok=True)

for html_file in html_dir.glob('*_3d_pointcloud_*.html'):
    scene_name = html_file.stem  # e.g., '0001_3d_pointcloud_pred_gt'
    
    plotly_config = extract_plotly_data(html_file)
    
    if plotly_config:
        # Save as JSON
        output_file = data_dir / f'{scene_name}.json'
        with open(output_file, 'w') as f:
            json.dump(plotly_config, f)
        
        print(f'Extracted: {scene_name}.json')