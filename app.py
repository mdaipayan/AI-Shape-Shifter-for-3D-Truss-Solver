import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core_solver import TrussSystem, Node, Member
from ai_optimizer import TrussOptimizer
from is_catalog import get_isa_catalog
import datetime
import os
from visualizer import draw_undeformed_geometry, draw_results_fbd

st.set_page_config(page_title="Professional Truss Suite (3D)", layout="wide")
st.title("🏗️ Professional Space Truss Analysis Developed by D Mandal")

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

fig = go.Figure()

def clear_results():
    if 'solved_truss' in st.session_state:
        del st.session_state['solved_truss']
    if 'report_data' in st.session_state:
        del st.session_state['report_data']
    if 'optimized_sections' in st.session_state:
        del st.session_state['optimized_sections']

# Initialize dynamic grouping text box state
if 'group_input_val' not in st.session_state:
    st.session_state['group_input_val'] = "1, 2, 3; 4, 5, 6"

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
                [4, 0.0, 50000.0, -100000.0]  
            ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])
            
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
                [1, 10000.0, 50000.0, -50000.0], [2, 0.0, 50000.0, -50000.0],
                [3, 10000.0, 0.0, 0.0], [6, 10000.0, 0.0, 0.0]
            ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])
            
            st.session_state['group_input_val'] = "1; 2, 3, 4, 5; 6, 7, 8, 9; 10, 11; 12, 13; 14, 15, 16, 17; 18, 19, 20, 21; 22, 23, 24, 25"
            clear_results()

    with col_btn3:
        if st.button("🏗️ Load 72-Bar"):
            # Parametric Generation of the 72-Bar Truss
            nodes = []
            # Base nodes (Level 0) - Fixed
            nodes.append([-1.5, 1.5, 0.0, 1, 1, 1])
            nodes.append([1.5, 1.5, 0.0, 1, 1, 1])
            nodes.append([1.5, -1.5, 0.0, 1, 1, 1])
            nodes.append([-1.5, -1.5, 0.0, 1, 1, 1])
            
            # Levels 1 to 4
            for i in range(1, 5):
                z = i * 1.5
                nodes.append([-1.5, 1.5, z, 0, 0, 0])
                nodes.append([1.5, 1.5, z, 0, 0, 0])
                nodes.append([1.5, -1.5, z, 0, 0, 0])
                nodes.append([-1.5, -1.5, z, 0, 0, 0])
            st.session_state['nodes_data'] = pd.DataFrame(nodes, columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
            
            members = []
            groups = []
            member_id = 1
            for t in range(4):
                B1, B2, B3, B4 = t*4+1, t*4+2, t*4+3, t*4+4
                T1, T2, T3, T4 = t*4+5, t*4+6, t*4+7, t*4+8
                
                # Verticals
                v_group = []
                for b, top in [(B1, T1), (B2, T2), (B3, T3), (B4, T4)]:
                    members.append([b, top, 0.005, 2e11])
                    v_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(v_group))
                
                # Horizontals
                h_group = []
                for n1, n2 in [(T1, T2), (T2, T3), (T3, T4), (T4, T1)]:
                    members.append([n1, n2, 0.005, 2e11])
                    h_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(h_group))
                
                # Face Diagonals
                fd_group = []
                for n1, n2 in [(B1, T2), (B2, T1), (B2, T3), (B3, T2), (B3, T4), (B4, T3), (B4, T1), (B1, T4)]:
                    members.append([n1, n2, 0.005, 2e11])
                    fd_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(fd_group))
                
                # Plan Diagonals
                pd_group = []
                for n1, n2 in [(T1, T3), (T2, T4)]:
                    members.append([n1, n2, 0.005, 2e11])
                    pd_group.append(str(member_id))
                    member_id += 1
                groups.append(", ".join(pd_group))
                
            st.session_state['members_data'] = pd.DataFrame(members, columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
            
            # Apply standard asymmetric loading at the top nodes
            st.session_state['loads_data'] = pd.DataFrame([
                [17, 50000.0, 50000.0, -25000.0],
                [18, 0.0, 0.0, -25000.0],
                [19, 0.0, 0.0, -25000.0],
                [20, 0.0, 0.0, -25000.0]
            ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])
            
            st.session_state['group_input_val'] = "; ".join(groups)
            clear_results()

    if 'nodes_data' not in st.session_state:
        st.session_state['nodes_data'] = pd.DataFrame(columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
        st.session_state['members_data'] = pd.DataFrame(columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
        st.session_state['loads_data'] = pd.DataFrame(columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])

    st.subheader("Nodes")
    node_df = st.data_editor(st.session_state['nodes_data'], num_rows="dynamic", key="nodes", on_change=clear_results)

    st.subheader("Members")
    member_df = st.data_editor(st.session_state['members_data'], num_rows="dynamic", key="members", on_change=clear_results)

    st.subheader("Nodal Loads")
    load_df = st.data_editor(st.session_state['loads_data'], num_rows="dynamic", key="loads", on_change=clear_results)

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
            ts = TrussSystem()
            node_map = {}
            valid_node_count = 0
            
            # 1. Parse Nodes
            for i, row in node_df.iterrows():
                if pd.isna(row.get('X')) or pd.isna(row.get('Y')) or pd.isna(row.get('Z')): continue
                valid_node_count += 1
                rx = int(row.get('Restrain_X', 0)) if not pd.isna(row.get('Restrain_X')) else 0
                ry = int(row.get('Restrain_Y', 0)) if not pd.isna(row.get('Restrain_Y')) else 0
                rz = int(row.get('Restrain_Z', 0)) if not pd.isna(row.get('Restrain_Z')) else 0
                
                n = Node(valid_node_count, float(row['X']), float(row['Y']), float(row['Z']), rx, ry, rz)
                n.user_id = i + 1 
                ts.nodes.append(n)
                node_map[i + 1] = n 
                
            # 2. Parse Members
            for i, row in member_df.iterrows():
                if pd.isna(row.get('Node_I')) or pd.isna(row.get('Node_J')): continue
                ni_val, nj_val = int(row['Node_I']), int(row['Node_J'])
                
                if ni_val not in node_map or nj_val not in node_map:
                    raise ValueError(f"Member M{i+1} references an invalid Node ID.")
                    
                E = float(row.get('E (N/sq.m)', 2e11)) if not pd.isna(row.get('E (N/sq.m)')) else 2e11
                A = float(row.get('Area(sq.m)', 0.01)) if not pd.isna(row.get('Area(sq.m)')) else 0.01
                ts.members.append(Member(i+1, node_map[ni_val], node_map[nj_val], E, A))
                
            # 3. Parse Loads
            for i, row in load_df.iterrows():
                if pd.isna(row.get('Node_ID')): continue
                node_id_val = int(row['Node_ID'])
                
                if node_id_val not in node_map:
                    raise ValueError(f"Load at row {i+1} references an invalid Node ID.")
                    
                target_node = node_map[node_id_val]
                fx = float(row.get('Force_X (N)', 0)) if not pd.isna(row.get('Force_X (N)')) else 0.0
                fy = float(row.get('Force_Y (N)', 0)) if not pd.isna(row.get('Force_Y (N)')) else 0.0
                fz = float(row.get('Force_Z (N)', 0)) if not pd.isna(row.get('Force_Z (N)')) else 0.0
                
                dof_x, dof_y, dof_z = target_node.dofs[0], target_node.dofs[1], target_node.dofs[2]
                
                ts.loads[dof_x] = ts.loads.get(dof_x, 0.0) + fx
                ts.loads[dof_y] = ts.loads.get(dof_y, 0.0) + fy
                ts.loads[dof_z] = ts.loads.get(dof_z, 0.0) + fz
            
            if not ts.nodes or not ts.members:
                raise ValueError("Incomplete model: Please define at least two valid nodes and one member.")
                
            # --- CHOOSE SOLVER BASED ON UI TOGGLE ---
            if analysis_type == "Linear Elastic (Standard)":
                ts.solve()
            else:
                with st.spinner(f"Running Non-Linear Newton-Raphson across {load_steps} increments..."):
                    ts.solve_nonlinear(load_steps=load_steps)
                    
            st.session_state['solved_truss'] = ts
            st.success(f"Analysis Complete using {analysis_type}!")
            
        except Exception as e:
            st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # NEW SECTION: IS 800 DISCRETE AI SIZE OPTIMIZATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧠 IS 800 Discrete AI Optimization")
    st.info("Utilizes an Evolutionary Algorithm to assign standard Indian Standard Equal Angles (ISA) from the SP(6) catalog. Evaluates structural stability against IS 800 column buckling curves.")
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        yield_stress_mpa = st.number_input("Steel Yield Stress (MPa)", value=250.0, step=10.0)
    with opt_col2:
        max_deflection_mm = st.number_input("Max Nodal Deflection (mm)", value=50.0, step=5.0)
        
    st.markdown("**Symmetry & Constructability (Member Grouping)**")
    st.caption("Enter comma-separated Member IDs to group them into identical sections. Separate groups with a semicolon (;).")
    
    # Text input mapped to session state
    grouping_input = st.text_input("Member Groups", key="group_input_val")
        
    if st.button("🚀 Run Discrete AI Optimization"):
        if 'solved_truss' not in st.session_state:
            st.warning("⚠️ Please run a standard 'Calculate Results' first to validate the base geometry.")
        else:
            try:
                parsed_groups = []
                for g in grouping_input.split(';'):
                    group = [int(x.strip()) for x in g.split(',') if x.strip()]
                    if group:
                        parsed_groups.append(group)
            except ValueError:
                st.error("❌ Invalid grouping format. Please use numbers separated by commas and semicolons.")
                parsed_groups = None

            if parsed_groups:
                with st.spinner("🧬 AI is testing thousands of discrete IS 800 angle combinations..."):
                    base_ts = st.session_state['solved_truss']
                    
                    try:
                        optimizer = TrussOptimizer(
                            base_truss=base_ts, 
                            member_groups=parsed_groups,
                            yield_stress=yield_stress_mpa * 1e6, 
                            max_deflection=max_deflection_mm / 1000.0 
                        )
                        
                        # NEW: Unpacking the 4th variable (history)
                        final_sections, final_weight, is_valid, history = optimizer.optimize(pop_size=20, max_gen=100) 
                        
                        if is_valid:
                            st.success("🎉 Discrete Optimization Converged Successfully!")
                            st.session_state['optimized_sections'] = final_sections
                            
                            orig_weight = sum([mbr.A * mbr.L * 7850 for mbr in base_ts.members])
                            weight_saved = orig_weight - final_weight
                            pct_saved = (weight_saved / orig_weight) * 100
                            
                            st.metric(
                                label="Total Optimized Steel Weight", 
                                value=f"{final_weight:.2f} kg", 
                                delta=f"-{weight_saved:.2f} kg ({pct_saved:.1f}% Lighter vs Baseline)", 
                                delta_color="inverse"
                            )
                            
                            # ---------------------------------------------------
                            # NEW: Plot the Academic Convergence Curve
                            # ---------------------------------------------------
                            st.markdown("### 📈 Evolutionary Convergence Curve")
                            st.caption("Validates algorithmic stability by tracking weight reduction across generations.")
                            
                            # Filter out massive penalty values from early random generations
                            clean_hist = [w for w in history if w < 1e6]
                            
                            if clean_hist:
                                fig_conv = go.Figure()
                                fig_conv.add_trace(go.Scatter(
                                    y=clean_hist,
                                    mode='lines+markers',
                                    name='Best Feasible Weight',
                                    line=dict(color='forestgreen', width=3),
                                    marker=dict(size=6, color='black')
                                ))
                                fig_conv.update_layout(
                                    xaxis_title="Generation (Epoch)",
                                    yaxis_title="Structural Weight (kg)",
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    height=350,
                                    plot_bgcolor="rgba(240, 240, 240, 0.5)"
                                )
                                st.plotly_chart(fig_conv, use_container_width=True)
                            
                            # ---------------------------------------------------
                            
                            results_df = pd.DataFrame({
                                "Member": [f"M{mbr.id}" for mbr in base_ts.members],
                                "Optimized IS 800 Section": [final_sections.get(mbr.id, "Error") for mbr in base_ts.members],
                            })
                            
                            st.dataframe(results_df)
                        else:
                            st.error("❌ Optimizer failed to find ANY catalog combination that satisfies the IS 800 constraints.")
                    except Exception as e:
                        st.error(f"Optimization Error: {e}")

    # The Apply Button
    if 'optimized_sections' in st.session_state:
        st.markdown("---")
        if st.button("✅ Apply Optimized Sections to Model"):
            df_m = st.session_state['members_data'].copy()
            catalog = get_isa_catalog()
            
            for i, row in df_m.iterrows():
                m_id = i + 1
                if m_id in st.session_state['optimized_sections']:
                    sec_name = st.session_state['optimized_sections'][m_id]
                    # Fetch the corresponding area in m^2 from the catalog
                    area_m2 = catalog[catalog['Designation'] == sec_name]['Area_m2'].values[0]
                    df_m.at[i, 'Area(sq.m)'] = area_m2
            
            # Save it back to the state and force a UI refresh
            st.session_state['members_data'] = df_m
            clear_results()
            st.success("Model updated! Scroll up and click 'Calculate Results' to view the new force distribution.")
            st.rerun()

    # ---------------------------------------------------------
    # NEW SECTION: PROFESSIONAL PDF REPORT GENERATION
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
                    
                    # Package AI data if optimization was run
                    opt_payload = None
                    if 'optimized_sections' in st.session_state:
                        # Recalculate weights quickly for the report
                        orig_w = sum([m.A * m.L * 7850 for m in base_ts.members])
                        from is_catalog import get_isa_catalog
                        cat = get_isa_catalog()
                        final_w = 0
                        for m in base_ts.members:
                            if m.id in st.session_state['optimized_sections']:
                                sec_name = st.session_state['optimized_sections'][m.id]
                                w_per_m = cat[cat['Designation'] == sec_name]['Weight_kg_m'].values[0]
                                final_w += m.L * w_per_m
                            else:
                                final_w += m.A * m.L * 7850
                                
                        opt_payload = {
                            'sections': st.session_state['optimized_sections'],
                            'orig_weight': orig_w,
                            'final_weight': final_w
                        }
                    
                    # Call the LaTeX compiler
                    pdf_bytes = generate_pdf_report(
                        ts_solved=base_ts, 
                        opt_data=opt_payload,
                        fig_base=fig_base_img, 
                        fig_res=fig_res_img,
                        scale_factor=current_scale, 
                        unit_label=current_unit
                    )
                    
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"Truss_Analysis_Report_{datetime.date.today().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        type="primary"
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
        if 'solved_truss' in st.session_state:
            ts = st.session_state['solved_truss']
            fig_res = draw_results_fbd(ts, scale_factor=current_scale, unit_label=current_unit)
            st.session_state['current_fig'] = fig_res 
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("👈 Input loads and click 'Calculate Results' to view the force diagram.")

# ---------------------------------------------------------
# NEW SECTION: THE "GLASS BOX" PEDAGOGICAL EXPLORER (3D)
# ---------------------------------------------------------
if 'solved_truss' in st.session_state:
    st.markdown("---")
    st.header("🎓 Educational Glass-Box: 3D DSM Intermediate Steps")
    
    ts = st.session_state['solved_truss']
    gb_tab1, gb_tab2, gb_tab3 = st.tabs(["📐 1. 3D Kinematics & Stiffness", "🧩 2. Global Assembly", "🚀 3. Displacements & Internal Forces"])
    
    with gb_tab1:
        st.subheader("Local Element Formulation (3D)")
        if ts.members: 
            mbr_opts = [f"Member {m.id}" for m in ts.members]
            sel_mbr = st.selectbox("Select Member to inspect kinematics and stiffness:", mbr_opts, key="gb_tab1")
            selected_id = int(sel_mbr.split(" ")[1])
            m = next((m for m in ts.members if m.id == selected_id), None)
            
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
                m = next((m for m in ts.members if m.id == selected_id), None)
                
                if m and hasattr(m, 'u_local') and m.u_local is not None:
                    st.latex(r"F_{axial} = \frac{EA}{L} \cdot (T \cdot u_{local})")
                    st.markdown("**Local Displacements ($u_{local}$):**")
                    st.dataframe(pd.DataFrame([m.u_local], columns=["u_ix", "u_iy", "u_iz", "u_jx", "u_jy", "u_jz"]).style.format("{:.6e}"))
                    st.success(f"**Calculated Axial Force:** {m.internal_force:.2f} N")
