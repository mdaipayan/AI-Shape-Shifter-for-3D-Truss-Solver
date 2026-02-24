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
from visualizer import draw_undeformed_geometry, draw_results_fbd

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
            
            st.session_state['shape_bounds_data'] = pd.DataFrame(columns=["Node_ID", "dX_min", "dX_max", "dY_min", "dY_max", "dZ_
