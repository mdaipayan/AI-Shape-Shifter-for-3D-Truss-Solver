import plotly.graph_objects as go
import pandas as pd
import numpy as np

def draw_undeformed_geometry(node_df, member_df, load_df, scale_factor=1000.0, unit_label="kN"):
    fig_base = go.Figure()
    node_errors, member_errors, load_errors = [], [], []

    # 1. Plot Nodes
    nx_vals, ny_vals, nz_vals, text_vals = [], [], [], []
    if not node_df.empty:
        for i, row in node_df.iterrows():
            try:
                if pd.isna(row.get('X')) or pd.isna(row.get('Y')) or pd.isna(row.get('Z')): continue
                nx_vals.append(float(row['X']))
                ny_vals.append(float(row['Y']))
                nz_vals.append(float(row['Z']))
                text_vals.append(f"Node {i+1}")
            except (ValueError, TypeError):
                node_errors.append(str(i+1))

        fig_base.add_trace(go.Scatter3d(
            x=nx_vals, y=ny_vals, z=nz_vals,
            mode='markers+text', text=text_vals, textposition="top center",
            marker=dict(size=6, color='black'), showlegend=False
        ))

    # 2. Plot Members and Member Labels
    mid_x, mid_y, mid_z, mbr_labels = [], [], [], []

    if not node_df.empty and not member_df.empty:
        for i, row in member_df.iterrows():
            try:
                ni, nj = int(row['Node_I'])-1, int(row['Node_J'])-1
                if ni not in node_df.index or nj not in node_df.index:
                    member_errors.append(str(i+1))
                    continue
                n1, n2 = node_df.loc[ni], node_df.loc[nj]

                x0, y0, z0 = float(n1['X']), float(n1['Y']), float(n1['Z'])
                x1, y1, z1 = float(n2['X']), float(n2['Y']), float(n2['Z'])

                # Draw the dashed line
                fig_base.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode='lines', line=dict(color='gray', width=4, dash='dash'), showlegend=False
                ))

                # Calculate midpoints for the Member ID labels
                mid_x.append((x0 + x1) / 2)
                mid_y.append((y0 + y1) / 2)
                mid_z.append((z0 + z1) / 2)
                mbr_labels.append(f"M{i+1}")

            except (ValueError, TypeError, IndexError, KeyError):
                member_errors.append(str(i+1))

        # Add the Member ID text tags at the midpoints
        if mbr_labels:
            fig_base.add_trace(go.Scatter3d(
                x=mid_x, y=mid_y, z=mid_z,
                mode='text', text=mbr_labels,
                textfont=dict(color='blue', size=11, family="Arial"),
                showlegend=False
            ))

    # Configure 3D Scene
    fig_base.update_layout(
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data' # Enforces 1:1:1 geometric scaling
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600
    )
    return fig_base, node_errors, member_errors, load_errors

def draw_results_fbd(ts, scale_factor=1000.0, unit_label="kN"):
    fig_res = go.Figure()

    # 1. Plot Nodes
    nx, ny, nz = [], [], []
    for node in ts.nodes:
        nx.append(node.x)
        ny.append(node.y)
        nz.append(node.z)

    fig_res.add_trace(go.Scatter3d(
        x=nx, y=ny, z=nz, mode='markers',
        marker=dict(size=6, color='black'), showlegend=False
    ))

    # 2. Plot Members with Colors and Force Labels
    mid_x, mid_y, mid_z, mid_text, mid_colors = [], [], [], [], []

    for mbr in ts.members:
        f = mbr.calculate_force()
        val_scaled = round(abs(f) / scale_factor, 2)

        # Color mapping
        if val_scaled < 0.01:
            color = "gray"
        else:
            color = "crimson" if f < 0 else "royalblue"

        x0, y0, z0 = mbr.node_i.x, mbr.node_i.y, mbr.node_i.z
        x1, y1, z1 = mbr.node_j.x, mbr.node_j.y, mbr.node_j.z

        # Draw Member Line
        fig_res.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines', line=dict(color=color, width=8), showlegend=False
        ))

        # Save midpoint data for text rendering (Now includes Member ID!)
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_z.append((z0 + z1) / 2)
        mid_text.append(f"M{mbr.id}: {val_scaled} {unit_label}")
        mid_colors.append(color)

    # Add force value text tags at midpoints
    fig_res.add_trace(go.Scatter3d(
        x=mid_x, y=mid_y, z=mid_z,
        mode='text', text=mid_text,
        textfont=dict(color=mid_colors, size=12, family="Arial Black"),
        showlegend=False
    ))

    # 3. Add 3D Annotations for Support Reactions
    scene_annotations = []
    for node in ts.nodes:
        if node.rx or node.ry or node.rz:
            rx_s = round(node.rx_val / scale_factor, 2) if node.rx else 0
            ry_s = round(node.ry_val / scale_factor, 2) if node.ry else 0
            rz_s = round(node.rz_val / scale_factor, 2) if node.rz else 0

            reac_text = f"Rx:{rx_s}<br>Ry:{ry_s}<br>Rz:{rz_s}"

            scene_annotations.append(dict(
                x=node.x, y=node.y, z=node.z,
                text=reac_text,
                showarrow=True, arrowhead=2, ax=30, ay=-40,
                font=dict(color="white", size=10),
                bgcolor="darkgreen"
            ))

    fig_res.update_layout(
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
            annotations=scene_annotations
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600
    )
    return fig_res

def draw_shape_optimization_overlay(ts, shifts):
    """Generates Figure 3 for the paper: 3D overlay of original vs optimized shape."""
    fig = go.Figure()

    # 1. Original Nodes & Members (Faded Gray)
    nx_base, ny_base, nz_base = [], [], []
    for n in ts.nodes:
        nx_base.append(n.x)
        ny_base.append(n.y)
        nz_base.append(n.z)

    fig.add_trace(go.Scatter3d(
        x=nx_base, y=ny_base, z=nz_base, mode='markers',
        marker=dict(size=4, color='lightgray'), name='Original Nodes'
    ))

    for m in ts.members:
        fig.add_trace(go.Scatter3d(
            x=[m.node_i.x, m.node_j.x], y=[m.node_i.y, m.node_j.y], z=[m.node_i.z, m.node_j.z],
            mode='lines', line=dict(color='lightgray', width=4, dash='solid'), showlegend=False
        ))

    # 2. Shifted Nodes & Members (Blue & Red)
    nx_opt, ny_opt, nz_opt = [], [], []
    shifted_nodes = {}
    for n in ts.nodes:
        dx = shifts[n.id]['dx'] if n.id in shifts else 0.0
        dy = shifts[n.id]['dy'] if n.id in shifts else 0.0
        dz = shifts[n.id]['dz'] if n.id in shifts else 0.0

        x_new, y_new, z_new = n.x + dx, n.y + dy, n.z + dz
        nx_opt.append(x_new)
        ny_opt.append(y_new)
        nz_opt.append(z_new)
        shifted_nodes[n.id] = (x_new, y_new, z_new)

        # Add red shift vectors if the node moved
        if abs(dx) > 1e-4 or abs(dy) > 1e-4 or abs(dz) > 1e-4:
            fig.add_trace(go.Scatter3d(
                x=[n.x, x_new], y=[n.y, y_new], z=[n.z, z_new],
                mode='lines', line=dict(color='red', width=5), name='Displacement Vector', showlegend=False
            ))

    fig.add_trace(go.Scatter3d(
        x=nx_opt, y=ny_opt, z=nz_opt, mode='markers',
        marker=dict(size=6, color='red'), name='Optimized Nodes'
    ))

    for m in ts.members:
        xi, yi, zi = shifted_nodes[m.node_i.id]
        xj, yj, zj = shifted_nodes[m.node_j.id]
        fig.add_trace(go.Scatter3d(
            x=[xi, xj], y=[yi, yj], z=[zi, zj],
            mode='lines', line=dict(color='royalblue', width=5), showlegend=False
        ))

    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(l=0, r=0, t=30, b=0), height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig
