import sys
sys.path.append('/home/llandsmeer/repos/llandsmeer/cerebellum-jax/')

import brainpy as bp
import brainpy.math as bm
from models.network import CerebellarNetwork

def generate_mermaid_text(model: bp.DynamicalSystem) -> str:
    mermaid_lines = ["graph TD"]
    added_nodes = set()
    def add_node(obj, name_hint):
        node_id = f"node_{id(obj)}"
        if node_id not in added_nodes:
            label = name_hint
            if hasattr(obj, 'num'):
                label = f"{name_hint} ({obj.num})"
            elif hasattr(obj, 'size'):
                 label = f"{name_hint} ({obj.size})"
            safe_label = label.replace('"', '#quot;')
            mermaid_lines.append(f'    {node_id}["{safe_label}"]')
            added_nodes.add(node_id)
        return node_id
    populations = {}
    synapses = []
    for name, obj in model.nodes(level=-1).items():
        if isinstance(obj, bp.dyn.NeuDyn):
             node_id = add_node(obj, name)
             populations[obj] = node_id
        elif isinstance(obj, bp.dyn.SynConn):
             synapses.append(obj)
    for syn in synapses:
        if hasattr(syn, 'pre') and hasattr(syn, 'post') and \
           syn.pre in populations and syn.post in populations:
            pre_node_id = populations[syn.pre]
            post_node_id = populations[syn.post]
            syn_name = ""
            for name, obj in model.nodes(level=-1).items():
                if obj is syn:
                    syn_name = name
                    break
            if syn_name:
                 mermaid_lines.append(f"    {pre_node_id} -->|{syn_name}| {post_node_id}")
            else:
                 mermaid_lines.append(f"    {pre_node_id} --> {post_node_id}")
        else:
            print(f"Warning: Could not find nodes for synapse: {syn}. Pre={getattr(syn,'pre', 'N/A')}, Post={getattr(syn,'post', 'N/A')}")
    return "\n".join(mermaid_lines)

model = CerebellarNetwork()
mermaid_code = generate_mermaid_text(model)

print("```mermaid")
print(mermaid_code)
print("```")

