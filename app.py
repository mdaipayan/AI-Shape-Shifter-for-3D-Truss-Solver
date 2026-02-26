import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core_solver import TrussSystem, Node, Member
from ai_optimizer import TrussOptimizer
from is_catalog import get_isa_catalog
import datetime
import os
import json
from visualizer import draw_undeformed_geometry, draw_results_fbd, draw_shape_optimization_overlay
st.set_page_config(page_title="Professional Truss Suite (3D)", layout="wide")
st.title("🏗️ Professional Space Truss Analysis Developed by D Mandal")

# ---------------------------------------------------------
# 1. INITIALIZE SESSION STATE (MUST BE AT THE TOP)
# ---------------------------------------------------------
if 'nodes_data' not in st.session_state:
    st.session_state['nodes_data'] = pd.DataFrame(columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
    st.session_state['members_data'] = pd.DataFrame(columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
    st.session_state['loads_data'] = pd.DataFrame(columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)", "Load_Case"])
    st.session_state['combos_data'] = pd.DataFrame([
        ["Serviceability (1.0DL + 1.0LL)", 1.0, 1.0],
        ["Ultimate Limit State (1.5DL + 1.5LL)", 1.5, 1.5]
    ], columns=["Combo_Name", "Factor_DL", "Factor_LL"])
    st.session_state['shape_bounds_data'] = pd.DataFrame(columns=["Node_ID", "dX_min", "dX_max", "dY_min", "dY_max", "dZ_min", "dZ_max"])
    
if 'group_input_val' not in st.session_state:
    st.session_state['group_input_val'] = "1, 2, 3; 4, 5, 6"

def clear_results():
    if 'solved_truss' in st.session_state:
        del st.session_state['solved_truss']
    if 'solved_combos' in st.session_state:
        del st.session_state['solved_combos']
    if 'report_data' in st.session_state:
        del st.session_state['report_data']
    if 'optimized_sections' in st.session_state:
        del st.session_state['optimized_sections']
    if 'optimized_shape' in st.session_state:
        del st.session_state['optimized_shape']

# ---------------------------------------------------------
# 2. SIDEBAR & SETTINGS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Display Settings")
st.sidebar.info("The solver engine calculates using base SI units (Newtons, meters). Use this setting to scale the visual output on the diagrams.")

force_display = st.sidebar.selectbox(
    "Force Display Unit", 
    options=["Newtons (N)", "Kilonewtons (kN)", "Meganewtons (MN)"], 
    index=1
)

unit_map = {
    "Newtons (N)": (1.0, "N"), 
    "Kilonewtons (kN)": (1000.0, "kN"), 
    "Meganewtons (MN)": (1000000.0, "MN")
}
current_scale, current_unit = unit_map[force_display]

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.sidebar.success("Memory Cache Cleared!")

# ---------------------------------------------------------
# SAVE / LOAD PROJECT (JSON)
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Project Management")

def export_project():
    project_data = {
        "nodes": st.session_state['nodes_data'].to_dict(orient='records'),
        "members": st.session_state['members_data'].to_dict(orient='records'),
        "loads": st.session_state['loads_data'].to_dict(orient='records'),
        "combos": st.session_state['combos_data'].to_dict(orient='records'),
        "shape_bounds": st.session_state['shape_bounds_data'].to_dict(orient='records'),
        "groups": st.session_state.get('group_input_val', "")
    }
    return json.dumps(project_data, indent=4)

st.sidebar.download_button(
    label="⬇️ Save Project (.json)",
    data=export_project(),
    file_name=f"Truss_Project_{datetime.date.today().strftime('%Y%m%d')}.json",
    mime="application/json"
)

uploaded_file = st.sidebar.file_uploader("⬆️ Load Project (.json)", type=["json"])
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        project_data = json.load(uploaded_file)
        
        st.session_state['nodes_data'] = pd.DataFrame(project_data['nodes'])
        st.session_state['members_data'] = pd.DataFrame(project_data['members'])
        st.session_state['loads_data'] = pd.DataFrame(project_data['loads'])
        st.session_state['combos_data'] = pd.DataFrame(project_data['combos'])
        st.session_state['shape_bounds_data'] = pd.DataFrame(project_data.get('shape_bounds', []))
        st.session_state['group_input_val'] = project_data.get('groups', "")
        
        clear_results() 
        st.sidebar.success("Project Loaded Successfully!")
        
        if st.sidebar.button("🔄 Refresh UI to View Loaded Data"):
            st.rerun()
            
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# ---------------------------------------------------------
# CACHED SOLVER ENGINE
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_structural_analysis(n_df, m_df, l_df, combo_factors, a_type, l_steps):
    ts = TrussSystem()
    node_map = {}
    valid_node_count = 0
    
    for i, row in n_df.iterrows():
        if pd.isna(row.get('X')) or pd.isna(row.get('Y')) or pd.isna(row.get('Z')): continue
        valid_node_count += 1
        rx = int(row.get('Restrain_X', 0)) if not pd.isna(row.get('Restrain_X')) else 0
        ry = int(row.get('Restrain_Y', 0)) if not pd.isna(row.get('Restrain_Y')) else 0
        rz = int(row.get('Restrain_Z', 0)) if not pd.isna(row.get('Restrain_Z')) else 0
        
        n = Node(valid_node_count, float(row['X']), float(row['Y']), float(row['Z']), rx, ry, rz)
        n.user_id = i + 1 
        ts.nodes.append(n)
        node_map[i + 1] = n 
        
    for i, row in m_df.iterrows():
        if pd.isna(row.get('Node_I')) or pd.isna(row.get('Node_J')): continue
        ni_val, nj_val = int(row['Node_I']), int(row['Node_J'])
        
        if ni_val not in node_map or nj_val not in node_map:
            raise ValueError(f"Member M{i+1} references an invalid Node ID.")
            
        E = float(row.get('E (N/sq.m)', 2e11)) if not pd.isna(row.get('E (N/sq.m)')) else 2e11
        A = float(row.get('Area(sq.m)', 0.01)) if not pd.isna(row.get('Area(sq.m)')) else 0.01
        ts.members.append(Member(i+1, node_map[ni_val], node_map[nj_val], E, A))
        
    for i, row in l_df.iterrows():
        if pd.isna(row.get('Node_ID')): continue
        node_id_val = int(row['Node_ID'])
        
        if node_id_val not in node_map:
            raise ValueError(f"Load at row {i+1} references an invalid Node ID.")
            
        target_node = node_map[node_id_val]
        
        case_name = str(row.get('Load_Case', 'DL')).strip()
        factor_col = f"Factor_{case_name}"
        factor = float(combo_factors.get(factor_col, 1.0)) 
        
        fx = float(row.get('Force_X (N)', 0)) * factor
        fy = float(row.get('Force_Y (N)', 0)) * factor
        fz = float(row.get('Force_Z (N)', 0)) * factor
        
        dof_x, dof_y, dof_z = target_node.dofs[0], target_node.dofs[1], target_node.dofs[2]
        
        ts.loads[dof_x] = ts.loads.get(dof_x, 0.0) + fx
        ts.loads[dof_y] = ts.loads.get(dof_y, 0.0) + fy
        ts.loads[dof_z] = ts.loads.get(dof_z, 0.0) + fz
    
    if not ts.nodes or not ts.members:
        raise ValueError("Incomplete model: Please define at least two valid nodes and one member.")
        
    if a_type == "Linear Elastic (Standard)":
        ts.solve()
    else:
        ts.solve_nonlinear(load_steps=l_steps)
            
    return ts

# ---------------------------------------------------------
# 3. MAIN UI LAYOUT
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Input Data")
    
    st.info("💡 **Benchmark Library:** Load standard geometries to test the solver and AI.")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔺 Load Tetrahedron"):
            st.session_state['nodes_data'] = pd.DataFrame([
                [0.0, 0.0, 0.0, 1, 1, 1],  
                [3.0, 0.0, 0.0, 0, 1, 1],  
                [1.5, 3.0, 0.0, 0, 0, 1],  
                [1.5, 1.5, 4.0, 0, 0, 0]   
            ], columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
            
            st.session_state['members_data'] = pd.DataFrame([
                [1, 2, 0.01, 2e11], [2, 3, 0.01, 2e11], [3, 1, 0.01, 2e11], 
                [1, 4, 0.01, 2e11], [2, 4, 0.01, 2e11], [3, 4, 0.01, 2e11]   
            ], columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
            
            st.session_state['loads_data'] = pd.DataFrame([
                [4, 0.0, 50000.0, -100000.0, "DL"]  
            ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)", "Load_Case"])
            
            st.session_state['combos_data'] = pd.DataFrame([
                ["Standard Combination", 1.0]
            ], columns=["Combo_Name", "Factor_DL"])
            
            st.session_state['shape_bounds_data'] = pd.DataFrame(columns=["Node_ID", "dX_min", "dX_max", "dY_min", "dY_max", "dZ_min", "dZ_max"])
            st.session_state['group_input_val'] = "1, 2, 3; 4, 5, 6"
            clear_results()

    with col_btn2:
        if st.button("🗼 Load 25-Bar"):
            st.session_state['nodes_data'] = pd.DataFrame([
                [-1.0, 0.0, 5.0, 0, 0, 0], [1.0, 0.0, 5.0, 0, 0, 0], 
                [-1.0, 1.0, 2.5, 0, 0, 0], [1.0, 1.0, 2.5, 0, 0, 0], 
                [1.0, -1.0, 2.5, 0, 0, 0], [-1.0, -1.0, 2.5, 0, 0, 0], 
                [-2.5, 2.5, 0.0, 1, 1, 1], [2.5, 2.5, 0.0, 1, 1, 1], 
                [2.5, -2.5, 0.0, 1, 1, 1], [-2.5, -2.5, 0.0, 1, 1, 1]
            ], columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
            
            st.session_state['members_data'] = pd.DataFrame([
                [1, 2, 0.005, 2e11], [1, 4, 0.005, 2e11], [2, 3, 0.005, 2e11], [1, 5, 0.005, 2e11], 
                [2, 6, 0.005, 2e11], [1, 3, 0.005, 2e11], [2, 4, 0.005, 2e11], [2, 5, 0.005, 2e11], 
                [1, 6, 0.005, 2e11], [3, 6, 0.005, 2e11], [4, 5, 0.005, 2e11], [3, 4, 0.005, 2e11], 
                [5, 6, 0.005, 2e11], [3, 10, 0.005, 2e11], [6, 7, 0.005, 2e11], [4, 9, 0.005, 2e11], 
                [5, 8, 0.005, 2e11], [3, 8, 0.005, 2e11], [4, 7, 0.005, 2e11], [6, 9, 0.005, 2e11], 
                [5, 10, 0.005, 2e11], [3, 7, 0.005, 2e11], [4, 8, 0.005, 2e11], [5, 9, 0.005, 2e11], 
                [6, 10, 0.005, 2e11]
            ], columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
            
            st.session_state['loads_data'] = pd.DataFrame([
                [1, 10000.0, 50000.0, -50000.0, "DL"], [2, 0.0, 50000.0, -50000.0, "DL"],
                [3, 10000.0, 0.0, 0.0, "WL"], [6, 10000.0, 0.0, 0.0, "WL"]
            ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)", "Load_Case"])
            
            st.session_state['combos_data'] = pd.DataFrame([
                ["Gravity (1.5DL)", 1.5, 0.0],
                ["Extreme Combo (1.2DL + 1.2WL)", 1.2, 1.2]
            ], columns=["Combo_Name", "Factor_DL", "Factor_WL"])
            
            st.session_state['shape_bounds_data'] = pd.DataFrame(columns=["Node_ID", "dX_min", "dX_max", "dY_min", "dY_max", "dZ_min", "dZ_max"])
            st.session_state['group_input_val'] = "1; 2, 3, 4, 5; 6, 7, 8, 9; 10, 11; 12, 13; 14, 15, 16, 17; 18, 19, 20, 21; 22, 23, 24, 25"
            clear_results()

    with col_btn3:
        if st.button("🏗️ Load 72-Bar"):
            nodes = []
            nodes.append([-1.5, 1.5, 0.0, 1, 1, 1])
            nodes.append([1.5, 1.5, 0.0, 1, 1, 1])
            nodes.append([1.5, -1.5, 0.0, 1, 1, 1])
            nodes.append([-1.5, -1.5, 0.0, 1, 1, 1])
            
            for i in range(1, 5):
                z = i * 1.5
                nodes.append([-1.5, 1.5, z, 0, 0, 0])
                nodes.append([1.5, 1.5, z, 0, 0, 0])
                nodes.append([1.5, -1.5, z, 0, 0, 0])
                nodes.append([-1.5, -1.5, z, 0, 0, 0])
            st.session_state['nodes_data'] = pd.DataFrame(nodes, columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
            
            members, groups = [], []
            member_id = 1
            for t in range(4):
                B1, B2, B3, B4 = t*4+1, t*4+2, t*4+3, t*4+4
                T1, T2, T3, T4 = t*4+5, t*4+6, t*4+7, t*4+8
                
                v_group = []
                for b, top in [(B1, T1), (B2, T2), (B3, T3), (B4, T4)]:
                    members.append([b, top, 0.005, 2e11])
                    v_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(v_group))
                
                h_group = []
                for n1, n2 in [(T1, T2), (T2, T3), (T3, T4), (T4, T1)]:
                    members.append([n1, n2, 0.005, 2e11])
                    h_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(h_group))
                
                fd_group = []
                for n1, n2 in [(B1, T2), (B2, T1), (B2, T3), (B3, T2), (B3, T4), (B4, T3), (B4, T1), (B1, T4)]:
                    members.append([n1, n2, 0.005, 2e11])
                    fd_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(fd_group))
                
                pd_group = []
                for n1, n2 in [(T1, T3), (T2, T4)]:
                    members.append([n1, n2, 0.005, 2e11])
                    pd_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(pd_group))
                
            st.session_state['members_data'] = pd.DataFrame(members, columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
            
            st.session_state['loads_data'] = pd.DataFrame([
                [17, 50000.0, 50000.0, -25000.0, "WL"],
                [18, 0.0, 0.0, -25000.0, "DL"],
                [19, 0.0, 0.0, -25000.0, "DL"],
                [20, 0.0, 0.0, -25000.0, "DL"]
            ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)", "Load_Case"])
            
            st.session_state['combos_data'] = pd.DataFrame([
                ["Serviceability (1.0DL + 1.0WL)", 1.0, 1.0],
                ["Ultimate (1.5DL + 1.5WL)", 1.5, 1.5],
                ["Dead Load Only (1.5DL)", 1.5, 0.0]
            ], columns=["Combo_Name", "Factor_DL", "Factor_WL"])
            
            st.session_state['shape_bounds_data'] = pd.DataFrame(columns=["Node_ID", "dX_min", "dX_max", "dY_min", "dY_max", "dZ_min", "dZ_max"])
            st.session_state['group_input_val'] = "; ".join(groups)
            clear_results()

    st.subheader("Nodes")
    node_df = st.data_editor(st.session_state['nodes_data'], num_rows="dynamic", key="nodes", on_change=clear_results)

    st.subheader("Members")
    member_df = st.data_editor(st.session_state['members_data'], num_rows="dynamic", key="members", on_change=clear_results)

    st.subheader("Nodal Loads (Base Cases)")
    st.info("Assign a string like 'DL', 'LL', or 'WL' to the Load_Case column.")
    load_df = st.data_editor(st.session_state['loads_data'], num_rows="dynamic", key="loads", on_change=clear_results)

    st.subheader("Load Combinations")
    st.info("Define multiplication factors. Column names must match `Factor_[Load_Case]` exactly.")
    combo_df = st.data_editor(st.session_state['combos_data'], num_rows="dynamic", key="combos", on_change=clear_results)

    st.markdown("---")
    st.subheader("⚙️ Solver Settings")
    
    analysis_type = st.radio(
        "Select Analysis Method:", 
        ["Linear Elastic (Standard)", "Non-Linear (Geometric P-Δ)"], 
        horizontal=True,
        on_change=clear_results
    )

    load_steps = 10
    if analysis_type == "Non-Linear (Geometric P-Δ)":
        load_steps = st.slider(
            "Newton-Raphson Load Steps", 
            min_value=5, max_value=50, value=10, step=5,
            help="Breaking the total load into smaller steps helps the non-linear solver converge."
        )
        st.info("💡 Non-linear analysis applies the load incrementally, updating the stiffness matrix as the geometry deforms.")
    
    if st.button("Calculate Results"):
        try:
            solved_combos = {}
            
            for idx, combo_row in combo_df.iterrows():
                combo_name = str(combo_row['Combo_Name'])
                combo_factors = combo_row.to_dict() 
                
                with st.spinner(f"Solving {combo_name}..."):
                    ts = run_structural_analysis(node_df, member_df, load_df, combo_factors, analysis_type, load_steps)
                    solved_combos[combo_name] = ts
                    
            st.session_state['solved_combos'] = solved_combos
            
            if solved_combos:
                st.session_state['solved_truss'] = list(solved_combos.values())[0] 
            
            st.success(f"Successfully analyzed {len(solved_combos)} load combinations!")
            
        except Exception as e:
            st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # IS 800 DISCRETE AI SIZE & SHAPE OPTIMIZATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧠 MINLP AI Optimization (Shape & Sizing)")
    st.info("Utilizes Differential Evolution to optimize standard IS 800 sections while simultaneously morphing node coordinates.")
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        yield_stress_mpa = st.number_input("Steel Yield Stress (MPa)", value=250.0, step=10.0)
    with opt_col2:
        max_deflection_mm = st.number_input("Max Nodal Deflection (mm)", value=50.0, step=5.0)
        
    st.markdown("**Sizing: Symmetry & Constructability (Member Grouping)**")
    grouping_input = st.text_input("Member Groups", key="group_input_val", on_change=clear_results)
    
    # NEW: Shape Optimization UI Table
    st.markdown("**Shape: Coordinate Bounds**")
    st.caption("Define allowable coordinate shifts (in meters) for specific nodes. Leave empty for pure sizing optimization.")
    shape_df = st.data_editor(st.session_state['shape_bounds_data'], num_rows="dynamic", key="shape_bounds", on_change=clear_results)
        
    if st.button("🚀 Run MINLP AI Optimization"):
        if 'solved_truss' not in st.session_state:
            st.warning("⚠️ Please run a standard 'Calculate Results' first to validate the base geometry.")
        else:
            try:
                # Parse Grouping
                parsed_groups = []
                for g in grouping_input.split(';'):
                    group = [int(x.strip()) for x in g.split(',') if x.strip()]
                    if group:
                        parsed_groups.append(group)
                        
                # Parse Shape Bounds
                shape_bounds_dict = {}
                for i, row in shape_df.iterrows():
                    if pd.isna(row.get('Node_ID')): continue
                    nid = int(row['Node_ID'])
                    dx_min = float(row.get('dX_min', 0.0)) if not pd.isna(row.get('dX_min')) else 0.0
                    dx_max = float(row.get('dX_max', 0.0)) if not pd.isna(row.get('dX_max')) else 0.0
                    dy_min = float(row.get('dY_min', 0.0)) if not pd.isna(row.get('dY_min')) else 0.0
                    dy_max = float(row.get('dY_max', 0.0)) if not pd.isna(row.get('dY_max')) else 0.0
                    dz_min = float(row.get('dZ_min', 0.0)) if not pd.isna(row.get('dZ_min')) else 0.0
                    dz_max = float(row.get('dZ_max', 0.0)) if not pd.isna(row.get('dZ_max')) else 0.0
                    
                    # Ensure minimum is actually less than or equal to maximum to prevent scipy crashing
                    if dx_min > dx_max: dx_min, dx_max = dx_max, dx_min
                    if dy_min > dy_max: dy_min, dy_max = dy_max, dy_min
                    if dz_min > dz_max: dz_min, dz_max = dz_max, dz_min
                    
                    shape_bounds_dict[nid] = [dx_min, dx_max, dy_min, dy_max, dz_min, dz_max]
                    
            except ValueError:
                st.error("❌ Invalid formatting in groups or shape bounds.")
                parsed_groups = None

            if parsed_groups:
                with st.spinner("🧬 AI is mutating geometries and testing IS 800 combinations..."):
                    base_ts = st.session_state['solved_truss']
                    
                    try:
                        solved_combos_dict = st.session_state['solved_combos']
                        
                        # Initialize the new TrussOptimizer with shape_bounds
                        optimizer = TrussOptimizer(
                            base_combos=list(solved_combos_dict.values()), 
                            is_nonlinear=(analysis_type == "Non-Linear (Geometric P-Δ)"),
                            load_steps=load_steps,
                            member_groups=parsed_groups,
                            shape_bounds=shape_bounds_dict,
                            yield_stress=yield_stress_mpa * 1e6, 
                            max_deflection=max_deflection_mm / 1000.0 
                        )
                        
                        # Catch the new 5-variable return tuple
                        final_sections, final_node_shifts, final_weight, is_valid, history = optimizer.optimize(pop_size=20, max_gen=100) 
                        
                        if is_valid:
                            st.success("🎉 MINLP Optimization Converged Successfully!")
                            st.session_state['optimized_sections'] = final_sections
                            st.session_state['optimized_shape'] = final_node_shifts
                            
                            orig_weight = sum([mbr.A * mbr.L * 7850 for mbr in base_ts.members])
                            weight_saved = orig_weight - final_weight
                            pct_saved = (weight_saved / orig_weight) * 100 if orig_weight > 0 else 0
                            
                            st.metric(
                                label="Total Optimized Steel Weight", 
                                value=f"{final_weight:.2f} kg", 
                                delta=f"-{weight_saved:.2f} kg ({pct_saved:.1f}% Lighter vs Baseline)", 
                                delta_color="inverse"
                            )
                            
                            st.markdown("### 📈 Evolutionary Convergence Curve")
                            st.caption("Validates algorithmic stability by tracking weight reduction across generations.")
                            
                            clean_hist = [w for w in history if w < 1e6]
                            
                            if clean_hist:
                                fig_conv = go.Figure()
                                fig_conv.add_trace(go.Scatter(
                                    y=clean_hist, mode='lines+markers', name='Best Feasible Weight',
                                    line=dict(color='forestgreen', width=3), marker=dict(size=6, color='black')
                                ))
                                fig_conv.update_layout(
                                    xaxis_title="Generation (Epoch)", yaxis_title="Structural Weight (kg)",
                                    margin=dict(l=0, r=0, t=10, b=0), height=350, plot_bgcolor="rgba(240, 240, 240, 0.5)"
                                )
                                st.plotly_chart(fig_conv, use_container_width=True)
                            
                            # Display Sizing Results
                            st.markdown("#### Sizing Output")
                            results_df = pd.DataFrame({
                                "Member": [f"M{mbr.id}" for mbr in base_ts.members],
                                "Optimized IS 800 Section": [final_sections.get(mbr.id, "Error") for mbr in base_ts.members],
                            })
                            st.dataframe(results_df)
                            
                            # Display Shape Results
                            if final_node_shifts:
                                st.markdown("#### Shape Output (Node Coordinate Shifts)")
                                shape_res_df = pd.DataFrame.from_dict(final_node_shifts, orient='index')
                                shape_res_df.index.name = "Node_ID"
                                st.dataframe(shape_res_df.style.format("{:.3f}"))
                                
                        else:
                            st.error("❌ Optimizer failed to find ANY sizing/shape combination that satisfies the IS 800 constraints.")
                    except Exception as e:
                        st.error(f"Optimization Error: {e}")

    # The Apply Button (Updated to handle shapes)
    if 'optimized_sections' in st.session_state and 'optimized_shape' in st.session_state:
        st.markdown("---")
        if st.button("✅ Apply Optimized Design to Model"):
            df_m = st.session_state['members_data'].copy()
            catalog = get_isa_catalog()
            
            # Apply Sizing
            for i, row in df_m.iterrows():
                m_id = i + 1
                if m_id in st.session_state['optimized_sections']:
                    sec_name = st.session_state['optimized_sections'][m_id]
                    area_m2 = catalog[catalog['Designation'] == sec_name]['Area_m2'].values[0]
                    df_m.at[i, 'Area(sq.m)'] = area_m2
            
            # Apply Shape
            df_n = st.session_state['nodes_data'].copy()
            shifts = st.session_state['optimized_shape']
            for i, row in df_n.iterrows():
                n_id = i + 1
                if n_id in shifts:
                    df_n.at[i, 'X'] += shifts[n_id]['dx']
                    df_n.at[i, 'Y'] += shifts[n_id]['dy']
                    df_n.at[i, 'Z'] += shifts[n_id]['dz']
            
            st.session_state['members_data'] = df_m
            st.session_state['nodes_data'] = df_n
            clear_results()
            st.success("Model updated! Scroll up and click 'Calculate Results' to view the new shape and force distribution.")
            st.rerun()

    # ---------------------------------------------------------
    # PROFESSIONAL PDF REPORT GENERATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📄 Export Documentation")
    
    if 'solved_truss' in st.session_state:
        from report_gen import generate_pdf_report
        
        if st.button("⚙️ Generate Professional PDF Report"):
            with st.spinner("Compiling LaTeX document (pdflatex)..."):
                try:
                    base_ts = st.session_state['solved_truss']
                    fig_base_img = st.session_state.get('base_fig', None)
                    fig_res_img = st.session_state.get('current_fig', None)
                    
                    opt_payload = None
                    if 'optimized_sections' in st.session_state:
                        orig_w = sum([mbr.A * mbr.L * 7850 for mbr in base_ts.members])
                        from is_catalog import get_isa_catalog
                        cat = get_isa_catalog()
                        final_w = 0
                        for mbr in base_ts.members:
                            if mbr.id in st.session_state['optimized_sections']:
                                sec_name = st.session_state['optimized_sections'][mbr.id]
                                w_per_m = cat[cat['Designation'] == sec_name]['Weight_kg_m'].values[0]
                                final_w += mbr.L * w_per_m
                            else:
                                final_w += mbr.A * mbr.L * 7850
                                
                        opt_payload = {
                            'sections': st.session_state['optimized_sections'],
                            'orig_weight': orig_w,
                            'final_weight': final_w
                        }
                    
                    pdf_bytes = generate_pdf_report(
                        ts_solved=base_ts, opt_data=opt_payload,
                        fig_base=fig_base_img, fig_res=fig_res_img,
                        scale_factor=current_scale, unit_label=current_unit
                    )
                    
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes, file_name=f"Truss_Analysis_Report_{datetime.date.today().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", type="primary"
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

with col2:
    st.header("2. 3D Model Visualization")
    tab1, tab2 = st.tabs(["🏗️ Undeformed Geometry", "📊 Structural Forces (Results)"])

    with tab1:
        if node_df.empty:
            st.info("👈 Start adding nodes in the Input Table (or click 'Load Benchmark Data') to build your geometry.")
        else:
            fig_base, node_errors, member_errors, load_errors = draw_undeformed_geometry(node_df, member_df, load_df, scale_factor=current_scale, unit_label=current_unit)
            
            if node_errors: st.warning(f"⚠️ Geometry Warning: Invalid data at Node row(s): {', '.join(node_errors)}.")
            if member_errors: st.warning(f"⚠️ Connectivity Warning: Cannot draw M{', M'.join(member_errors)}.")
            
            st.session_state['base_fig'] = fig_base 
            st.plotly_chart(fig_base, use_container_width=True)

    with tab2:
        if 'solved_combos' in st.session_state and st.session_state['solved_combos']:
            combo_names = list(st.session_state['solved_combos'].keys())
            selected_combo_vis = st.selectbox("👁️ View Results for Load Combination:", combo_names)
            
            ts_to_view = st.session_state['solved_combos'][selected_combo_vis]
            
            fig_res = draw_results_fbd(ts_to_view, scale_factor=current_scale, unit_label=current_unit)
            st.session_state['current_fig'] = fig_res 
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("👈 Input loads and click 'Calculate Results' to view the force diagram.")

# ---------------------------------------------------------
# NEW SECTION: THE "GLASS BOX" PEDAGOGICAL EXPLORER (3D)
# ---------------------------------------------------------
if 'solved_combos' in st.session_state and st.session_state['solved_combos']:
    st.markdown("---")
    st.header("🎓 Educational Glass-Box: 3D DSM Intermediate Steps")
    
    combo_names = list(st.session_state['solved_combos'].keys())
    selected_combo_gb = st.selectbox("📐 Inspect Matrix Math for Load Combination:", combo_names, key="gb_selector")
    ts = st.session_state['solved_combos'][selected_combo_gb]
    
    gb_tab1, gb_tab2, gb_tab3 = st.tabs(["📐 1. 3D Kinematics & Stiffness", "🧩 2. Global Assembly", "🚀 3. Displacements & Internal Forces"])
    
    with gb_tab1:
        st.subheader("Local Element Formulation (3D)")
        if ts.members: 
            mbr_opts = [f"Member {mbr.id}" for mbr in ts.members]
            sel_mbr = st.selectbox("Select Member to inspect kinematics and stiffness:", mbr_opts, key="gb_tab1")
            selected_id = int(sel_mbr.split(" ")[1])
            m = next((mbr for mbr in ts.members if mbr.id == selected_id), None)
            
            colA, colB = st.columns([1, 2])
            with colA:
                st.markdown("**Member Kinematics**")
                st.write(f"- **Length ($L$):** `{m.L:.4f} m`")
                st.write(f"- **Dir. Cosine X ($l$):** `{m.l:.4f}`")
                st.write(f"- **Dir. Cosine Y ($m$):** `{m.m:.4f}`")
                st.write(f"- **Dir. Cosine Z ($n$):** `{m.n:.4f}`")
                
                st.markdown("**Transformation Vector ($T$):**")
                st.dataframe(pd.DataFrame([m.T_vector], columns=["-l", "-m", "-n", "l", "m", "n"]).style.format("{:.4f}"))
            
            with colB:
                st.markdown("**6x6 Global Element Stiffness Matrix ($k_{global}$)**")
                df_k = pd.DataFrame(m.k_global_matrix)
                st.dataframe(df_k.style.format("{:.2e}"))

    with gb_tab2:
        st.subheader("System Partitioning & Assembly")
        colC, colD = st.columns(2)
        with colC:
            st.markdown("**Degree of Freedom (DOF) Mapping**")
            st.write(f"- **Free DOFs ($f$):** `{ts.free_dofs}`")
            st.write(f"- **Active Load Vector ($F_f$)**")
            st.dataframe(pd.DataFrame(ts.F_reduced, columns=["Force"]).style.format("{:.2e}"))

        with colD:
            with st.expander("View Full Unpartitioned Global Matrix ($K_{global}$)", expanded=True):
                st.dataframe(pd.DataFrame(ts.K_global).style.format("{:.2e}"))
            with st.expander("View Reduced Stiffness Matrix ($K_{ff}$)", expanded=False):
                st.dataframe(pd.DataFrame(ts.K_reduced).style.format("{:.2e}"))

    with gb_tab3:
        st.subheader("Solving the System & Extracting Forces")
        colE, colF = st.columns(2)
        with colE:
            st.markdown("**1. Global Displacement Vector ($U_{global}$)**")
            if hasattr(ts, 'U_global') and ts.U_global is not None:
                st.dataframe(pd.DataFrame(ts.U_global, columns=["Displacement (m)"]).style.format("{:.6e}"))
                
        with colF:
            st.markdown("**2. Internal Force Extraction**")
            if ts.members:
                sel_mbr_force = st.selectbox("Select Member to view Force Extraction:", mbr_opts, key="gb_tab3")
                selected_id = int(sel_mbr_force.split(" ")[1])
                m = next((mbr for mbr in ts.members if mbr.id == selected_id), None)
                
                if m and hasattr(m, 'u_local') and m.u_local is not None:
                    st.latex(r"F_{axial} = \frac{EA}{L} \cdot (T \cdot u_{local})")
                    st.markdown("**Local Displacements ($u_{local}$):**")
                    st.dataframe(pd.DataFrame([m.u_local], columns=["u_ix", "u_iy", "u_iz", "u_jx", "u_jy", "u_jz"]).style.format("{:.6e}"))
                    st.success(f"**Calculated Axial Force:** {m.internal_force:.2f} N")
