"""
Generate 6 Advanced MIA notebooks with different attack/seed configurations.
Each notebook is a copy of run_all_experiments_dual.ipynb with only the
Advanced MIA cell modified.
"""

import json
import os

# Configuration for each notebook (9 notebooks, one per attack-seed pair)
NOTEBOOKS = [
    {"name": "advanced_mia_lira_s1", "attack": "lira", "seeds": [1]},
    {"name": "advanced_mia_lira_s2", "attack": "lira", "seeds": [2]},
    {"name": "advanced_mia_lira_s3", "attack": "lira", "seeds": [3]},
    {"name": "advanced_mia_shadow_s1", "attack": "shadow", "seeds": [1]},
    {"name": "advanced_mia_shadow_s2", "attack": "shadow", "seeds": [2]},
    {"name": "advanced_mia_shadow_s3", "attack": "shadow", "seeds": [3]},
    {"name": "advanced_mia_quantile_s1", "attack": "quantile", "seeds": [1]},
    {"name": "advanced_mia_quantile_s2", "attack": "quantile", "seeds": [2]},
    {"name": "advanced_mia_quantile_s3", "attack": "quantile", "seeds": [3]},
]

# Template for the modified Advanced MIA cell
def get_advanced_mia_cell(attack, seeds):
    seeds_str = str(seeds)
    return [
        f"# ========== ADVANCED MIA EXPERIMENTS ==========\n",
        f"# This notebook runs: {attack.upper()} with seeds {seeds}\n",
        f"# Estimated time: ~{len(seeds) * 9 * 10}h ({len(seeds)} seeds x 9 methods x ~10h each)\n",
        f"\n",
        f"ATTACK = \"{attack}\"\n",
        f"ATTACK_SEEDS = {seeds_str}\n",
        f"\n",
        f"for exp_config in CONFIG.get('advanced_mia_experiments', []):\n",
        f"    name = exp_config['name']\n",
        f"    \n",
        f"    print(f\"\\n{{'#'*80}}\")\n",
        f"    print(f\"# ADVANCED MIA EXPERIMENT: {{name}}\")\n",
        f"    print(f\"# Attack: {{ATTACK}}, Seeds: {{ATTACK_SEEDS}}\")\n",
        f"    print(f\"{{'#'*80}}\")\n",
        f"    \n",
        f"    for seed in ATTACK_SEEDS:\n",
        f"        for method_config in METHOD_CONFIGS:\n",
        f"            try:\n",
        f"                run_main_new_metrics(exp_config, seed, method_config, attack=ATTACK)\n",
        f"            except Exception as e:\n",
        f"                print(f\"ERROR: {{e}}\")\n",
        f"    \n",
        f"    # Compile results\n",
        f"    output_folder = os.path.join(OUTPUT_BASE, f\"Results_{{name}}_{{ATTACK.capitalize()}}\")\n",
        f"    csv_path = run_convert_csv(output_folder)\n",
        f"    if os.path.exists(csv_path):\n",
        f"        run_calculate_miau(csv_path)\n",
        f"    print(f\"{{ATTACK.capitalize()}} MIA results saved to: {{csv_path}}\")"
    ]

def main():
    # Read the original notebook
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, "run_all_experiments_dual.ipynb")
    
    with open(source_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the Advanced MIA cell index (the one with "ADVANCED MIA EXPERIMENTS")
    advanced_mia_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '# ========== ADVANCED MIA EXPERIMENTS ==========' in source:
                advanced_mia_cell_idx = i
                break
    
    if advanced_mia_cell_idx is None:
        print("ERROR: Could not find Advanced MIA cell in notebook")
        return
    
    print(f"Found Advanced MIA cell at index {advanced_mia_cell_idx}")
    
    # Generate each notebook
    for config in NOTEBOOKS:
        # Deep copy the notebook
        new_notebook = json.loads(json.dumps(notebook))
        
        # Replace the Advanced MIA cell
        new_notebook['cells'][advanced_mia_cell_idx]['source'] = get_advanced_mia_cell(
            config['attack'], config['seeds']
        )
        
        # Update the title cell (first markdown cell)
        title_update = f"\n\n**This notebook runs: {config['attack'].upper()} attack with seeds {config['seeds']}**\n"
        if new_notebook['cells'][0]['cell_type'] == 'markdown':
            new_notebook['cells'][0]['source'].append(title_update)
        
        # Write the new notebook
        output_path = os.path.join(script_dir, f"{config['name']}.ipynb")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_notebook, f, indent=2)
        
        est_hours = len(config['seeds']) * 9 * 10
        print(f"Created: {config['name']}.ipynb ({config['attack']}, seeds={config['seeds']}, ~{est_hours}h)")
    
    print(f"\nGenerated {len(NOTEBOOKS)} notebooks!")
    print("\nSummary:")
    print("| Notebook | Attack | Seeds | Est. Time |")
    print("|----------|--------|-------|-----------|")
    for config in NOTEBOOKS:
        est_hours = len(config['seeds']) * 9 * 10
        print(f"| {config['name']}.ipynb | {config['attack']} | {config['seeds']} | ~{est_hours}h |")

if __name__ == "__main__":
    main()
