import os
import tempfile
import datetime
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('helvetica', 'B', 15)
        # Title
        self.cell(0, 10, 'Professional 3D Space Truss Analysis', ln=True, align='C')
        self.set_font('helvetica', 'I', 11)
        self.cell(0, 8, 'Developed by Mr. D Mandal | Assistant Professor, KITS Ramtek', ln=True, align='C')
        self.cell(0, 8, f'Date: {datetime.date.today().strftime("%B %d, %Y")}', ln=True, align='C')
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(ts_solved, opt_data=None, fig_base=None, fig_res=None, scale_factor=1000.0, unit_label="kN"):
    """
    Generates a professional PDF report natively using FPDF2 (Open Source Python PDF).
    """
    pdf = PDF()
    pdf.add_page()
    
    temp_dir = tempfile.mkdtemp()
    
    # --- 1. Structural Visualization ---
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '1. Structural Visualization', ln=True)
    pdf.ln(2)
    
    if fig_base:
        base_img_path = os.path.join(temp_dir, "geometry.png")
        fig_base.write_image(base_img_path, engine="kaleido", width=800, height=500, scale=2)
        pdf.image(base_img_path, w=170)
        pdf.set_font('helvetica', 'I', 10)
        pdf.cell(0, 6, 'Figure 1: Undeformed 3D Truss Geometry', ln=True, align='C')
        pdf.ln(5)
        
    if fig_res:
        res_img_path = os.path.join(temp_dir, "results.png")
        fig_res.write_image(res_img_path, engine="kaleido", width=800, height=500, scale=2)
        pdf.image(res_img_path, w=170)
        pdf.set_font('helvetica', 'I', 10)
        pdf.cell(0, 6, 'Figure 2: Structural Forces and Reactions Diagram', ln=True, align='C')
        pdf.ln(5)
        
    # --- 2. Linear Static Analysis Results ---
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '2. Linear Static Analysis Results', ln=True)
    
    # 2.1 Nodal Displacements
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '2.1 Nodal Displacements', ln=True)
    pdf.set_font('helvetica', 'B', 10)
    
    col_widths = [30, 50, 50, 50]
    headers = ['Node ID', 'Ux (m)', 'Uy (m)', 'Uz (m)']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align='C')
    pdf.ln()
    
    pdf.set_font('helvetica', '', 10)
    for node in ts_solved.nodes:
        ux = ts_solved.U_global[node.dofs[0]] if ts_solved.U_global is not None else 0
        uy = ts_solved.U_global[node.dofs[1]] if ts_solved.U_global is not None else 0
        uz = ts_solved.U_global[node.dofs[2]] if ts_solved.U_global is not None else 0
        pdf.cell(col_widths[0], 8, str(node.id), border=1, align='C')
        pdf.cell(col_widths[1], 8, f"{ux:.6e}", border=1, align='C')
        pdf.cell(col_widths[2], 8, f"{uy:.6e}", border=1, align='C')
        pdf.cell(col_widths[3], 8, f"{uz:.6e}", border=1, align='C')
        pdf.ln()
    pdf.ln(5)
    
    # 2.2 Internal Axial Forces
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '2.2 Internal Axial Forces', ln=True)
    pdf.set_font('helvetica', 'B', 10)
    
    col_widths = [30, 40, 60, 50]
    headers = ['Member', 'Nodes', f'Axial Force ({unit_label})', 'Nature']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align='C')
    pdf.ln()
    
    pdf.set_font('helvetica', '', 10)
    for m in ts_solved.members:
        force = m.internal_force / scale_factor
        nature = "Tension" if force > 1e-6 else ("Compression" if force < -1e-6 else "Zero Force")
        
        pdf.cell(col_widths[0], 8, f"M{m.id}", border=1, align='C')
        pdf.cell(col_widths[1], 8, f"{m.node_i.id} - {m.node_j.id}", border=1, align='C')
        pdf.cell(col_widths[2], 8, f"{abs(force):.2f}", border=1, align='C')
        
        if nature == "Tension":
            pdf.set_text_color(0, 0, 255) # Blue text
        elif nature == "Compression":
            pdf.set_text_color(255, 0, 0) # Red text
        else:
            pdf.set_text_color(100, 100, 100) # Gray text
            
        pdf.cell(col_widths[3], 8, nature, border=1, align='C')
        pdf.set_text_color(0, 0, 0) # Reset to black
        pdf.ln()
        
    pdf.ln(5)

    # --- 3. AI Optimization Section ---
    if opt_data and 'sections' in opt_data:
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, '3. IS 800 Discrete AI Optimization', ln=True)
        
        orig_wt = opt_data.get('orig_weight', 0)
        final_wt = opt_data.get('final_weight', 0)
        saved = orig_wt - final_wt
        pct = (saved / orig_wt * 100) if orig_wt > 0 else 0
        
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, '3.1 Optimization Metrics', ln=True)
        pdf.set_font('helvetica', '', 11)
        pdf.cell(0, 8, f"- Original Steel Weight: {orig_wt:.2f} kg", ln=True)
        pdf.cell(0, 8, f"- Optimized Steel Weight: {final_wt:.2f} kg", ln=True)
        pdf.cell(0, 8, f"- Material Saved: {saved:.2f} kg ({pct:.1f}%)", ln=True)
        pdf.ln(5)
        
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, '3.2 Final IS 800 Assigned Sections', ln=True)
        
        pdf.set_font('helvetica', 'B', 10)
        col_widths = [50, 100]
        pdf.cell(col_widths[0], 8, 'Member ID', border=1, align='C')
        pdf.cell(col_widths[1], 8, 'Optimized Section (SP 6)', border=1, align='C')
        pdf.ln()
        
        pdf.set_font('helvetica', '', 10)
        for m_id, sec in opt_data['sections'].items():
            pdf.cell(col_widths[0], 8, f"M{m_id}", border=1, align='C')
            pdf.cell(col_widths[1], 8, sec, border=1, align='C')
            pdf.ln()

    # Output to raw bytes for Streamlit download
    return bytes(pdf.output())
