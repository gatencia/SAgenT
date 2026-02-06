import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

import numpy as np
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.db.schema import EntryRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_color_for_family(family: str, family_colors: Dict[str, str]) -> str:
    if family not in family_colors:
        # Generate a random vibrant hex color
        family_colors[family] = f"#{random.randint(0, 0xFFFFFF):06x}"
    return family_colors[family]

def build_visualization(db_path: str = "Denabase/my_database", 
                      output_file: str = "denabase_graph.html",
                      semantic_threshold: float = 0.85,
                      structural_threshold: float = 0.85,
                      max_nodes: int = 800,
                      max_edges_per_node: int = 2):
    """
    Creates an interactive knowledge graph of Denabase entries (optimized for size & UI).
    """
    logger.info(f"Opening Denabase at {db_path}...")
    db = DenaBase.open(db_path)
    
    # 1. Collect and Filter Entries
    all_entries = db.store.load_entries()
    logger.info(f"Total entries in DB: {len(all_entries)}")
    
    def get_source(e):
        s = e.meta.source or e.meta.user_meta.get("source")
        return s
    
    # Prioritize Agent-generated and Verified content
    agent_entries = [e for e in all_entries if get_source(e) == "agent"]
    other_entries = [e for e in all_entries if get_source(e) != "agent"]
    
    # Sub-select for visualization performance
    random.shuffle(other_entries)
    selected_entries = agent_entries + other_entries[:max(0, max_nodes - len(agent_entries))]
    selected_ids = {e.id for e in selected_entries}
    id_to_record = {entry.id: entry for entry in selected_entries}
    
    logger.info(f"Visualizing {len(selected_entries)} nodes ({len(agent_entries)} from agent).")
    
    # 2. Extract Embeddings (Subsampled)
    # Structural
    struct_map = {}
    if hasattr(db.index, "vectors") and db.index.vectors is not None:
        for eid, vec in zip(db.index.ids, db.index.vectors):
            if eid in selected_ids:
                struct_map[eid] = vec
    
    # Semantic
    semantic_map = {}
    if hasattr(db.nl_index, "vectors") and db.nl_index.vectors is not None:
        for eid, vec in zip(db.nl_index.ids, db.nl_index.vectors):
            if eid in selected_ids:
                semantic_map[eid] = vec

    # 3. Initialize Graph
    G = nx.Graph()
    family_colors = {}
    
    for entry in selected_entries:
        base_color = get_color_for_family(entry.meta.family, family_colors)
        source = get_source(entry)
        
        # Style
        border_color = "#333333"
        border_width = 1
        if source == "agent":
            border_color = "#FFD700" # Gold
            border_width = 4
        
        is_verified = entry.meta.user_meta.get("is_verified", True)
        node_color = base_color
        if not is_verified:
            node_color = "#999999"

        size = 15
        if entry.stats_summary:
            n_vars = entry.stats_summary.get("n_vars", entry.stats_summary.get("num_vars", 0))
            size = 15 + min(30, np.sqrt(n_vars))

        # Better Labeling Heuristic
        # If problem_id is "row_...", try to use nl_text summary
        label_text = entry.meta.problem_id
        if label_text.startswith("row_") or label_text.startswith("auto_") or len(label_text) > 20:
            if entry.meta.nl_text:
                # First 4 words or 22 chars
                words = entry.meta.nl_text.split()
                # Remove common starter words
                if words and words[0].lower() in ["a", "the", "generate", "create"]:
                    words = words[1:]
                
                candidate = " ".join(words[:4])
                if len(candidate) > 25:
                    candidate = candidate[:22] + "..."
                if candidate:
                    label_text = candidate

        prefix = "🤖 " if source == "agent" else "📄 "
        label = f"{prefix}{label_text}"
        
        source_str = f"🤖 Agent Generated" if source == "agent" else "📥 Context Ingested"
        verified_str = "✅ Verified" if is_verified else "❌ Unverified"
        
        title = (
            f"<div style='font-family: sans-serif; padding: 8px;'>"
            f"<b style='color: {base_color}'>{entry.meta.family.upper()}</b><br>"
            f"<b>Problem:</b> {entry.meta.problem_id}<br><hr>"
            f"<b>Source:</b> {source_str}<br>"
            f"<b>Status:</b> {verified_str}<br>"
            f"<b>Vars/Clauses:</b> {entry.stats_summary.get('n_vars','?')}/{entry.stats_summary.get('n_clauses','?')}<br><hr>"
            f"<i>{entry.meta.nl_text[:150] if entry.meta.nl_text else 'No desc'}...</i>"
            f"</div>"
        )
        
        description_safe = (entry.meta.nl_text or "No description available.").replace('"', '&quot;')

        G.add_node(entry.id, 
                   label=label, 
                   title=title, 
                   color={'background': node_color, 'border': border_color, 'highlight': {'background': base_color, 'border': '#ff0000'}},
                   size=size,
                   borderWidth=border_width,
                   # Embed detailed data for JS click handler
                   custom_data={
                       "problem_id": entry.meta.problem_id,
                       "family": entry.meta.family,
                       "source": source,
                       "verified": is_verified,
                       "vars": entry.stats_summary.get("n_vars", 0),
                       "clauses": entry.stats_summary.get("n_clauses", 0),
                       "description": description_safe,
                       "full_label": label_text
                   })

    # 4. Optimized Edge Creation (k-NN style)
    def add_top_edges(vector_map, threshold, edge_color, edge_label):
        if not vector_map: return
        
        ids = list(vector_map.keys())
        matrix = np.array([vector_map[eid] for eid in ids])
        
        if len(matrix) < 2: return
        
        logger.info(f"Computing {edge_label} similarities...")
        sims = cosine_similarity(matrix)
        
        for i, eid_a in enumerate(ids):
            # Get top indices excluding self
            row = sims[i]
            top_indices = np.argsort(row)[-(max_edges_per_node + 1):-1][::-1]
            
            for j in top_indices:
                score = row[j]
                if score > threshold:
                    eid_b = ids[j]
                    if G.has_edge(eid_a, eid_b):
                        G[eid_a][eid_b]['title'] += f"<br>{edge_label}: {score:.1%}"
                        G[eid_a][eid_b]['color']['color'] = "#d35400" # Stronger orange/purple for double link
                        G[eid_a][eid_b]['width'] = max(G[eid_a][eid_b]['width'], float(score)*4)
                        # Add raw data for JS
                        if 'sims' not in G[eid_a][eid_b]: G[eid_a][eid_b]['sims'] = []
                        G[eid_a][eid_b]['sims'].append(f"{edge_label}: {score:.1%}")
                    else:
                        G.add_edge(eid_a, eid_b, 
                                  weight=float(score),
                                  title=f"{edge_label}: {score:.1%}",
                                  color={"color": edge_color, "opacity": 0.5},
                                  width=float(score)*2,
                                  sims=[f"{edge_label}: {score:.1%}"])

    add_top_edges(semantic_map, semantic_threshold, "#3498db", "Semantic Analog")
    add_top_edges(struct_map, structural_threshold, "#e67e22", "Structural Similarity")

    # 5. Export
    logger.info(f"Generating HTML: {output_file}")
    net = Network(height="100vh", width="100%", bgcolor="#fcfcfc", font_color="black", notebook=False)
    net.from_nx(G)
    
    net.set_options("""
    var options = {
      "nodes": { "shadow": true, "shape": "dot" },
      "edges": { "smooth": { "type": "continuous" } },
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -80, "centralGravity": 0.01, "springLength": 120, "springConstant": 0.05 },
        "solver": "forceAtlas2Based",
        "timestep": 0.25,
        "stabilization": { "iterations": 150 }
      },
      "interaction": { "hover": true, "navigationButtons": true }
    }
    """)
    
    net.save_graph(output_file)
    
    # 6. Inject Sidebar and JS (Post-processing)
    logger.info("Injecting custom UI...")
    
    stats_html = f"""
    <div id="stats-panel" style="position: absolute; top: 10px; right: 10px; width: 300px; background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: 'Segoe UI', sans-serif; z-index: 1000; overflow-y: auto; max-height: 90vh;">
        <h2 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; font-size: 18px;">Denabase Explorer</h2>
        
        <div style="margin-bottom: 15px;">
            <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px;">Network Stats</div>
            <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{len(selected_entries)} Nodes</div>
            <div style="font-size: 13px; color: #7f8c8d;">{G.number_of_edges()} Connections</div>
        </div>
        
        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
            <div style="flex: 1; background: #fff3cd; padding: 10px; border-radius: 6px; border-left: 4px solid #f1c40f;">
                <div style="font-size: 11px; color: #856404; font-weight: bold;">AGENT GENERATED</div>
                <div style="font-size: 20px; font-weight: bold; color: #856404;">{len(agent_entries)}</div>
            </div>
            <div style="flex: 1; background: #d4edda; padding: 10px; border-radius: 6px; border-left: 4px solid #28a745;">
                <div style="font-size: 11px; color: #155724; font-weight: bold;">VERIFIED</div>
                <div style="font-size: 20px; font-weight: bold; color: #155724;">{len([e for e in selected_entries if e.meta.user_meta.get("is_verified", True)])}</div>
            </div>
        </div>
        
        <h3 style="font-size: 14px; margin: 15px 0 10px; color: #34495e; text-transform: uppercase; border-bottom: 1px solid #eee; padding-bottom: 5px;">Active Selection</h3>
        <div id="node-details" style="font-size: 13px; color: #555;">
            <p style="color: #95a5a6; font-style: italic; text-align: center; padding: 20px;">Click a node to view detailed metrics.</p>
        </div>
        
        <div id="connections-panel" style="display: none; margin-top: 15px;">
            <h4 style="font-size: 13px; margin: 0 0 5px; color: #34495e; font-weight: 600;">Related Problems</h4>
            <ul id="conn-list" style="list-style: none; padding: 0; margin: 0; font-size: 12px; max-height: 200px; overflow-y: auto; border: 1px solid #eee; border-radius: 4px;">
            </ul>
        </div>
        
        <div style="margin-top: 20px; font-size: 10px; color: #bdc3c7; text-align: center; border-top: 1px solid #f0f0f0; padding-top: 10px;">
            Generated by Denabase v0.2 | Interactive Graph
        </div>
    </div>
    
    <script type="text/javascript">
    // Wait for network to be ready
    setTimeout(function() {{
        if (typeof network !== 'undefined') {{
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    var meta = node.custom_data || {{}};
                    
                    var html = "<div style='animation: fadeIn 0.3s;'>";
                    html += "<div style='font-weight: bold; font-size: 16px; color: #2c3e50; margin-bottom: 5px;'>" + (meta.full_label || meta.problem_id) + "</div>";
                    html += "<div style='margin-bottom: 8px;'><span style='background: " + node.color.background + "; color: #fff; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;'>" + meta.family + "</span></div>";
                    
                    var sourceIcon = meta.source === 'agent' ? '🤖' : '📥';
                    html += "<div style='margin-bottom: 8px; display: flex; align-items: center; gap: 5px;'>";
                    html += "<span style='font-weight: 600; font-size: 12px;'>Source:</span> <span style='background: #f8f9fa; padding: 2px 6px; border-radius: 4px; border: 1px solid #eee;'>" + sourceIcon + " " + (meta.source === 'agent' ? 'Agent' : 'Ingested') + "</span>";
                    html += "</div>";
                    
                    html += "<div style='margin-bottom: 12px; font-size: 12px; line-height: 1.5; color: #555; background: #f9f9f9; padding: 8px; border-radius: 4px; border-left: 3px solid #3498db;'>" + meta.description + "</div>";
                    
                    html += "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; text-align: center;'>";
                    html += "<div style='background: #fff; border: 1px solid #eee; padding: 5px; border-radius: 4px;'><div style='color:#7f8c8d; font-size:10px;'>VARIABLES</div><div style='font-weight:bold;'>" + meta.vars + "</div></div>";
                    html += "<div style='background: #fff; border: 1px solid #eee; padding: 5px; border-radius: 4px;'><div style='color:#7f8c8d; font-size:10px;'>CLAUSES</div><div style='font-weight:bold;'>" + meta.clauses + "</div></div>";
                    html += "</div></div>";
                    
                    document.getElementById("node-details").innerHTML = html;
                    
                    // Edges
                    var connList = document.getElementById("conn-list");
                    connList.innerHTML = "";
                    var connectedEdges = network.getConnectedEdges(nodeId);
                    var hasConns = false;
                    
                    connectedEdges.forEach(function(edgeId) {{
                        var edge = edges.get(edgeId);
                        var connectedNodeId = (edge.from === nodeId) ? edge.to : edge.from;
                        var connectedNode = nodes.get(connectedNodeId);
                        var label = connectedNode.label || connectedNode.custom_data.problem_id;
                        
                        var sims = edge.sims || ["Sim: " + edge.width];
                        var simHtml = sims.map(s => "<span style='display:inline-block; background:#eef2f5; color:#555; padding:1px 5px; border-radius:3px; font-size:10px; margin-right:3px;'>" + s + "</span>").join("");
                        
                        var li = document.createElement("li");
                        li.style.padding = "8px";
                        li.style.borderBottom = "1px solid #f0f0f0";
                        li.innerHTML = "<div style='font-weight:600; font-size:12px; margin-bottom:2px;'>" + label + "</div>" + simHtml;
                        connList.appendChild(li);
                        hasConns = true;
                    }});
                    
                    if (!hasConns) {{
                         connList.innerHTML = "<li style='padding:8px; color:#999; font-style:italic;'>No significant connections found.</li>";
                    }}
                    
                    document.getElementById("connections-panel").style.display = "block";
                }} else {{
                    document.getElementById("node-details").innerHTML = "<p style='color: #95a5a6; font-style: italic; text-align: center; padding: 20px;'>Click a node to view detailed metrics.</p>";
                    document.getElementById("connections-panel").style.display = "none";
                }}
            }});
        }}
    }}, 1000);
    </script>
    <style>
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(5px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: #f1f1f1; }}
    ::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #bbb; }}
    </style>
    """
    
    # Read, inject, write back
    with open(output_file, "r") as f:
        content = f.read()
        
    if "</body>" in content:
        content = content.replace("</body>", f"{stats_html}</body>")
    else:
        content += stats_html
        
    with open(output_file, "w") as f:
        f.write(content)

    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="Denabase/my_database")
    parser.add_argument("--out", type=str, default="denabase_graph.html")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--edges", type=int, default=2)
    
    args = parser.parse_args()
    build_visualization(db_path=args.db, output_file=args.out, max_nodes=args.nodes, max_edges_per_node=args.edges)
