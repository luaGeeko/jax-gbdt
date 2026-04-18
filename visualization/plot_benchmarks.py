import pandas as pd
import plotly.graph_objects as go
import os

def generate_automated_thesis_plots(csv_path="results/thesis_benchmarks.csv"):
    if not os.path.exists(csv_path):
        print(f"🚨 Could not find {csv_path}. Run the benchmark script first!")
        return

    # 1. Load the Data
    print(f"Loading benchmark data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Define the exact configurations we want to plot
    configs = [
        {"hw_match": "cpu", "batch": 10000, "label": "Mac CPU<br>(Batch: 10k)"},
        {"hw_match": "cuda|gpu", "batch": 500000, "label": "NVIDIA GPU<br>(Batch: 500k)"},
        {"hw_match": "tpu", "batch": 500000, "label": "Google TPU<br>(Batch: 500k)"}
    ]

    labels = []
    baseline_ips, no_branch_ips, iterative_ips, dense_ips, laplacian_ips = [], [], [], [], []

    # 3. Extract the exact numbers from the DataFrame
    for conf in configs:
        labels.append(conf["label"])
        
        subset = df[(df['Hardware'].str.contains(conf['hw_match'], case=False, na=False, regex=True)) & 
                    (df['Batch_Size'] == conf['batch'])]
        
        if not subset.empty:
            # Baseline (XGBoost)
            baseline_val = subset['XGBoost_IPS'].max() / 1_000_000
            baseline_ips.append(round(baseline_val, 2))
            
            # Phase 1: No Branch (Discrete)
            nb_val = subset[subset['Method'] == 'no_branch']['JAX_IPS'].max()
            no_branch_ips.append(round(nb_val / 1_000_000, 2) if pd.notna(nb_val) and nb_val > 0 else 0.001)

            # Phase 2: Iterative (Sparse SpMV)
            iter_val = subset[subset['Method'] == 'soft_iterative']['JAX_IPS'].max()
            iterative_ips.append(round(iter_val / 1_000_000, 2) if pd.notna(iter_val) and iter_val > 0 else 0.001)
            
            # Phase 2: Dense Option B (The Holy Grail)
            dense_val = subset[subset['Method'] == 'soft_dense']['JAX_IPS'].max()
            dense_ips.append(round(dense_val / 1_000_000, 2) if pd.notna(dense_val) and dense_val > 0 else 0.001)
            
            # Phase 3: Laplacian Dense (Check for 50k batch on TPU specifically)
            if "tpu" in conf['hw_match']:
                lap_subset = df[(df['Hardware'].str.contains('tpu', case=False, na=False, regex=True)) & 
                                (df['Batch_Size'] == 50000) & 
                                (df['Method'] == 'laplacian_dense')]
                lap_val = lap_subset['JAX_IPS'].max()
                laplacian_ips.append(round(lap_val / 1_000_000, 3) if pd.notna(lap_val) and lap_val > 0 else 0.001)
            else:
                laplacian_ips.append(0.001) # N/A for CPU/GPU in this specific chart layout
        else:
            # Placeholders if hardware hasn't been benchmarked yet
            baseline_ips.append(0.001)
            no_branch_ips.append(0.001)
            iterative_ips.append(0.001)
            dense_ips.append(0.001)
            laplacian_ips.append(0.001)

    # --- Plotly Figure Setup ---
    fig = go.Figure()

    # XGBoost Baseline (Gray)
    fig.add_trace(go.Bar(name='XGBoost (C++ Baseline)', x=labels, y=baseline_ips, marker_color='#7f8c8d', 
                         text=[f"{v}M" if v > 0.01 else "N/A" for v in baseline_ips], textposition='auto'))
    
    # Phase 1: No Branch (Blue)
    fig.add_trace(go.Bar(name='Phase 1: No Branch (Discrete)', x=labels, y=no_branch_ips, marker_color='#3498db', 
                         text=[f"{v}M" if v > 0.01 else "N/A" for v in no_branch_ips], textposition='auto'))

    # Phase 2: Iterative (Red)
    fig.add_trace(go.Bar(name='Phase 2: Soft Iterative (Gather)', x=labels, y=iterative_ips, marker_color='#e74c3c', 
                         text=[f"{v}M" if v > 0.01 else "N/A" for v in iterative_ips], textposition='auto'))
    
    # Phase 2: Dense Option B (Green)
    fig.add_trace(go.Bar(name='Phase 2: Soft Dense (Solved)', x=labels, y=dense_ips, marker_color='#2ecc71', 
                         text=[f"{v}M" if v > 0.01 else "N/A" for v in dense_ips], textposition='auto', textfont=dict(color='white', size=14)))
    
    # Phase 3: Laplacian Dense (Purple)
    fig.add_trace(go.Bar(name='Phase 3: Laplacian (OOM bottleneck)', x=labels, y=laplacian_ips, marker_color='#9b59b6', 
                         text=[f"{v}M (50k batch)" if v > 0.01 else "N/A" for v in laplacian_ips], textposition='auto'))

    # --- Aesthetics ---
    fig.update_layout(
        title={
            'text': "<b>Architectural Ablation Study: Throughput Across Hardware</b><br><sup>Decision Forest Math Equivalencies in JAX (Log Scale)</sup>", 
            'y':0.95, 'x':0.5, 'xanchor': 'center'
        },
        yaxis_type="log",
        yaxis_title="Throughput (Millions of Inferences/Sec) - Log Scale",
        barmode='group', 
        plot_bgcolor='rgba(240, 242, 245, 1)', 
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(b=100) # Give extra room for the wider legend
    )

    filename = "thesis_hardware_benchmark_automated.html"
    fig.write_html(filename)
    print(f"✅ Interactive Plotly Dashboard saved to {os.getcwd()}/{filename}")
    
    # Try to open it
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(filename))
    except:
        pass

if __name__ == "__main__":
    generate_automated_thesis_plots()