import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def generate_advanced_thesis_plots(csv_path="results/thesis_benchmarks.csv"):
    if not os.path.exists(csv_path):
        print(f"Could not find {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    
    # Clean up hardware names for the legend
    def clean_hw(name):
        name = str(name).lower()
        if 'cpu' in name: return 'CPU'
        if 't4' in name or 'gpu' in name: return 'NVIDIA GPU'
        if 'tpu' in name: return 'Google TPU'
        return name
        
    df['HW_Clean'] = df['Hardware'].apply(clean_hw)

    # ==========================================
    # PLOT 1: Relative Speedup Scaling Curve
    # ==========================================
    fig1 = go.Figure()
    
    # We only care about plotting the successful, highly-optimized methods here
    target_methods = {
        'no_branch': 'Phase 1: No Branch',
        'soft_dense': 'Phase 2: Dense Tensor'
    }
    
    colors = {'CPU': '#7f8c8d', 'NVIDIA GPU': '#2ecc71', 'Google TPU': '#3498db'}
    dash_styles = {'no_branch': 'dash', 'soft_dense': 'solid'}

    for hw in ['CPU', 'NVIDIA GPU', 'Google TPU']:
        for method_key, method_name in target_methods.items():
            subset = df[(df['HW_Clean'] == hw) & (df['Method'] == method_key)].sort_values('Batch_Size')
            
            if not subset.empty:
                fig1.add_trace(go.Scatter(
                    x=subset['Batch_Size'],
                    y=subset['Speedup_Multiplier'],
                    mode='lines+markers',
                    name=f"{hw} ({method_name})",
                    line=dict(color=colors[hw], dash=dash_styles[method_key], width=3),
                    marker=dict(size=10)
                ))

    # Add a baseline reference line at 1.0x (XGBoost)
    fig1.add_shape(type="line", x0=0, y0=1, x1=df['Batch_Size'].max(), y1=1,
                   line=dict(color="red", width=2, dash="dot"))
    fig1.add_annotation(x=df['Batch_Size'].max() * 0.9, y=1.2, text="XGBoost Baseline (1.0x)", showarrow=False, font=dict(color="red"))

    fig1.update_layout(
        title="<b>Hardware Scaling: Relative Speedup vs XGBoost Baseline</b><br><sup>How JAX performance scales exponentially with batch size</sup>",
        xaxis_title="Batch Size",
        yaxis_title="Speedup Multiplier (Higher is Better)",
        yaxis_type="log", # Log scale helps show the massive GPU/TPU multipliers clearly
        plot_bgcolor='rgba(240, 242, 245, 1)',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    fig1.write_html("thesis_plot_speedup_curve.html")
    print(f"Speedup Curve saved to thesis_plot_speedup_curve.html")

    # ==========================================
    # PLOT 2: The XLA Compilation "Cold Start" Tax
    # ==========================================
    fig2 = go.Figure()
    
    # Get the max compilation time for each method across hardware
    comp_df = df.groupby(['HW_Clean', 'Method'])['Compilation_Time_sec'].max().reset_index()
    
    for hw in ['CPU', 'NVIDIA GPU', 'Google TPU']:
        subset = comp_df[comp_df['HW_Clean'] == hw]
        if not subset.empty:
            fig2.add_trace(go.Bar(
                name=hw,
                x=subset['Method'],
                y=subset['Compilation_Time_sec'],
                marker_color=colors[hw],
                text=[f"{v:.2f}s" for v in subset['Compilation_Time_sec']],
                textposition='auto'
            ))

    fig2.update_layout(
        title="<b>The XLA Compiler Tax: Cold-Start Compilation Time</b><br><sup>Overhead required to fuse decision tree topologies into silicon</sup>",
        xaxis_title="Mathematical Topology (JAX Kernel)",
        yaxis_title="Compilation Time (Seconds)",
        barmode='group',
        plot_bgcolor='rgba(240, 242, 245, 1)',
        paper_bgcolor='white'
    )
    
    fig2.write_html("thesis_plot_compilation_overhead.html")
    print(f"Compilation Overhead Plot saved to thesis_plot_compilation_overhead.html")

if __name__ == "__main__":
    generate_advanced_thesis_plots()