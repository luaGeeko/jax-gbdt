import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

def generate_batch_scaling_plots(csv_path="results/thesis_benchmarks.csv"):
    if not os.path.exists(csv_path):
        print(f"Could not find {csv_path}. Run the benchmark script first!")
        return

    print(f"Loading benchmark data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Automatically detect the batch sizes you ran
    batch_sizes = sorted(df['Batch_Size'].unique())
    if not batch_sizes:
        print("No data found in the CSV.")
        return

    # Setup a 2x2 grid for the 4 batch sizes
    # If you have more or fewer batch sizes, Plotly handles it gracefully if configured well, 
    # but we assume 4 based on your benchmark script [10000, 50000, 100000, 500000]
    rows = 2
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=[f"Batch Size: {b:,} Samples" for b in batch_sizes],
        vertical_spacing=0.15, 
        horizontal_spacing=0.05
    )

    # Define the methods and their consistent colors across all plots
    methods = [
        {'col_name': 'XGBoost Baseline', 'df_val': 'baseline', 'color': '#7f8c8d'},
        {'col_name': 'Phase 1: No Branch', 'df_val': 'no_branch', 'color': '#3498db'},
        {'col_name': 'Phase 2: Iterative (Gather)', 'df_val': 'soft_iterative', 'color': '#e74c3c'},
        {'col_name': 'Phase 2: Dense (Solved)', 'df_val': 'soft_dense', 'color': '#2ecc71'},
        {'col_name': 'Phase 3: Laplacian Dense', 'df_val': 'laplacian_dense', 'color': '#9b59b6'}
    ]

    # Universal X-axis hardware labels based on your CSV entries
    hardwares = ['cpu', 'Tesla T4', 'TPU']
    hw_labels = ['Colab CPU', 'NVIDIA T4', 'Google TPU']

    # Loop through each batch size and populate its specific subplot
    for i, batch in enumerate(batch_sizes):
        if i >= 4: break # Limit to 2x2 grid if you ran more than 4 batch sizes
        
        row = (i // cols) + 1
        col = (i % cols) + 1
        b_df = df[df['Batch_Size'] == batch]
        
        for m_idx, method in enumerate(methods):
            y_vals = []
            texts = []
            
            for hw in hardwares:
                # Use regex=True so 'TPU' matches 'TPU v5 lite', 'Tesla T4' matches, etc.
                hw_df = b_df[b_df['Hardware'].str.contains(hw, case=False, na=False, regex=True)]
                
                if method['df_val'] == 'baseline':
                    val = hw_df['XGBoost_IPS'].max() if not hw_df.empty else np.nan
                else:
                    val = hw_df[hw_df['Method'] == method['df_val']]['JAX_IPS'].max()
                
                if pd.notna(val) and val > 0:
                    y_vals.append(val / 1_000_000) # Convert to Millions
                    texts.append(f"{val/1_000_000:.1f}M")
                else:
                    y_vals.append(0.001) # 0.001 keeps the Log scale from crashing
                    if hw_df.empty:
                        texts.append("N/A") # Hardware hasn't been run yet
                    else:
                        texts.append("OOM") # Hardware crashed (Laplacian limit)
            
            # Only show the legend for the very first subplot so it isn't duplicated 4 times
            show_legend = True if i == 0 else False 
            
            fig.add_trace(go.Bar(
                name=method['col_name'],
                x=hw_labels,
                y=y_vals,
                marker_color=method['color'],
                text=texts,
                textposition='auto',
                showlegend=show_legend,
                textfont=dict(color='white' if method['df_val'] == 'soft_dense' else 'black')
            ), row=row, col=col)
            
        # Enforce Log scale for all subplots
        fig.update_yaxes(type="log", title_text="Throughput (Millions IPS)" if col == 1 else "", row=row, col=col)

    # --- Global Layout Aesthetics ---
    fig.update_layout(
        title={
            'text': "<b>Hardware Scaling Dynamics: Throughput vs. Batch Size</b><br><sup>Evaluating JAX Compiler Fusions across CPU, GPU, and TPU Architectures (Log Scale)</sup>", 
            'x': 0.5, 'xanchor': 'center'
        },
        barmode='group',
        height=900, # Taller height to accommodate the 2x2 grid beautifully
        plot_bgcolor='rgba(240, 242, 245, 1)', 
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(size=14)),
        margin=dict(t=120, b=100) # Extra padding
    )
    
    # Add subtle gridlines to all subplots
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(189, 195, 199, 0.5)')

    filename = "thesis_batch_scaling_grid.html"
    fig.write_html(filename)
    print(f"2x2 Subplot Dashboard saved to {os.getcwd()}/{filename}")
    
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(filename))
    except:
        pass

if __name__ == "__main__":
    generate_batch_scaling_plots()