import os
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from config import Config
from src.utils import logger

class ReportGenerator:
    def __init__(self, db_session):
        self.db = db_session

    def generate_pdf_report(self, start_date, end_date):
        """
        Generates a professional PDF report containing summary statistics
        and details of all violations within the specified date range.
        """
        from src.database import Violation, Camera, DetectionLog
        
        # Ensure reports directory exists
        os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
        
        filename = f"safety_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.pdf"
        file_path = os.path.join(Config.REPORTS_FOLDER, filename)
        
        # 1. Fetch data from DB
        violations = self.db.query(Violation).filter(
            Violation.timestamp >= start_date,
            Violation.timestamp <= end_date
        ).order_by(Violation.timestamp.desc()).all()
        
        # Fetch stats
        total_violations = len(violations)
        total_cameras = self.db.query(Camera).filter(Camera.status == 'active').count()
        
        # Calculate average compliance
        logs = self.db.query(DetectionLog).filter(
            DetectionLog.timestamp >= start_date,
            DetectionLog.timestamp <= end_date
        ).all()
        
        avg_compliance = 100.0
        if logs:
            avg_compliance = sum([log.compliance_percentage for log in logs]) / len(logs)
            
        # Group violations by type for statistics
        v_types = {}
        for v in violations:
            v_types[v.violation_type] = v_types.get(v.violation_type, 0) + 1

        # 2. Build PDF Document
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=30, leftMargin=30, topMargin=40, bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        # Custom premium style palette
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor('#1E293B'), # Deep navy Slate
            spaceAfter=15
        )
        
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor('#0F172A'),
            spaceBefore=15,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#334155'),
            leading=14
        )

        header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            textColor=colors.white
        )

        cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor('#334155'),
            leading=12
        )

        elements = []
        
        # Title and Header
        elements.append(Paragraph("Safety & Hygiene Monitoring Report", title_style))
        date_range_str = f"Reporting Period: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
        elements.append(Paragraph(date_range_str, body_style))
        elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", cell_style))
        elements.append(Spacer(1, 15))
        
        # 3. KPI Highlights Cards Table
        kpi_data = [
            [
                Paragraph("<b>Active Cameras</b>", cell_style),
                Paragraph("<b>Total Safety Incidents</b>", cell_style),
                Paragraph("<b>Avg. Compliance Index</b>", cell_style)
            ],
            [
                Paragraph(f"<font size=14 color='#2563EB'><b>{total_cameras}</b></font>", cell_style),
                Paragraph(f"<font size=14 color='#DC2626'><b>{total_violations}</b></font>", cell_style),
                Paragraph(f"<font size=14 color='#16A34A'><b>{avg_compliance:.1f}%</b></font>", cell_style)
            ]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 20))
        
        # 4. Incident Distribution Table
        elements.append(Paragraph("Incident Distribution Summary", section_style))
        dist_data = [[Paragraph("<b>Violation Category</b>", header_style), Paragraph("<b>Occurrence Count</b>", header_style)]]
        
        for v_type, count in v_types.items():
            dist_data.append([
                Paragraph(v_type, cell_style),
                Paragraph(str(count), cell_style)
            ])
            
        if not v_types:
            dist_data.append([Paragraph("No violations recorded", cell_style), Paragraph("0", cell_style)])
            
        dist_table = Table(dist_data, colWidths=[4.0*inch, 2.6*inch])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(dist_table)
        elements.append(Spacer(1, 20))
        
        # 5. Incident Log Table (Includes Screenshots)
        elements.append(Paragraph("Detailed Incident Logs", section_style))
        
        # Columns: Time, Camera, Violation, Conf, Screenshot
        log_data = [[
            Paragraph("<b>Timestamp (UTC)</b>", header_style),
            Paragraph("<b>Camera Name</b>", header_style),
            Paragraph("<b>Violation Type</b>", header_style),
            Paragraph("<b>Conf</b>", header_style),
            Paragraph("<b>Evidence Capture</b>", header_style)
        ]]
        
        for v in violations:
            cam_name = v.camera.name if v.camera else f"Cam {v.camera_id}"
            time_str = v.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            conf_str = f"{v.confidence * 100:.1f}%"
            
            # Embed image if path exists
            img_flowable = Paragraph("No Image", cell_style)
            if v.screenshot_path and os.path.exists(v.screenshot_path):
                try:
                    # Resize to fit table: 100 width, 75 height
                    img_flowable = RLImage(v.screenshot_path, width=100, height=75)
                except Exception as e:
                    logger.error(f"Error loading image into PDF: {e}")
                    
            log_data.append([
                Paragraph(time_str, cell_style),
                Paragraph(cam_name, cell_style),
                Paragraph(f"<font color='#DC2626'><b>{v.violation_type}</b></font>", cell_style),
                Paragraph(conf_str, cell_style),
                img_flowable
            ])
            
        if total_violations == 0:
            log_data.append([
                Paragraph("No incidents recorded in this window.", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style)
            ])
            
        # Total printable width is around 7.5 inches (540 points)
        log_table = Table(log_data, colWidths=[1.3*inch, 1.4*inch, 1.6*inch, 0.7*inch, 1.6*inch])
        log_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        
        elements.append(log_table)
        
        # Build Document
        doc.build(elements)
        logger.info(f"PDF report successfully created at: {file_path}")
        return file_path

    def generate_excel_report(self, start_date, end_date):
        """
        Generates an Excel workbook with sheets for incident lists and summary stats.
        """
        from src.database import Violation, Camera, DetectionLog
        
        os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
        filename = f"safety_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.xlsx"
        file_path = os.path.join(Config.REPORTS_FOLDER, filename)
        
        # 1. Fetch data
        violations = self.db.query(Violation).filter(
            Violation.timestamp >= start_date,
            Violation.timestamp <= end_date
        ).all()
        
        # 2. Build Incidents dataframe
        v_list = []
        for v in violations:
            v_list.append({
                'Incident ID': v.id,
                'Timestamp (UTC)': v.timestamp,
                'Camera Name': v.camera.name if v.camera else f"Cam {v.camera_id}",
                'Department': v.camera.department if v.camera else "Unknown",
                'Violation Type': v.violation_type,
                'Confidence': v.confidence,
                'Status': 'Resolved' if v.resolved else 'Pending',
                'Notes': v.resolution_notes or ''
            })
            
        df_incidents = pd.DataFrame(v_list)
        if df_incidents.empty:
            df_incidents = pd.DataFrame(columns=['Incident ID', 'Timestamp (UTC)', 'Camera Name', 'Department', 'Violation Type', 'Confidence', 'Status', 'Notes'])

        # 3. Build Compliance logs dataframe
        logs = self.db.query(DetectionLog).filter(
            DetectionLog.timestamp >= start_date,
            DetectionLog.timestamp <= end_date
        ).all()
        
        log_list = []
        for l in logs:
            log_list.append({
                'Timestamp (UTC)': l.timestamp,
                'Camera Name': l.camera.name if l.camera else f"Cam {l.camera_id}",
                'Worker Count': l.worker_count,
                'Compliance Index (%)': l.compliance_percentage
            })
        df_compliance = pd.DataFrame(log_list)
        if df_compliance.empty:
            df_compliance = pd.DataFrame(columns=['Timestamp (UTC)', 'Camera Name', 'Worker Count', 'Compliance Index (%)'])

        # 4. Grouped stats sheet
        stats_summary = []
        if not df_incidents.empty:
            type_counts = df_incidents['Violation Type'].value_counts()
            for v_type, count in type_counts.items():
                stats_summary.append({
                    'Category': 'Violation Occurrences',
                    'Item': v_type,
                    'Metric': count
                })
                
            camera_counts = df_incidents['Camera Name'].value_counts()
            for cam, count in camera_counts.items():
                stats_summary.append({
                    'Category': 'Incidents by Camera',
                    'Item': cam,
                    'Metric': count
                })
        
        if df_compliance.empty:
            avg_comp = 100.0
        else:
            avg_comp = df_compliance['Compliance Index (%)'].mean()
            
        stats_summary.append({
            'Category': 'General Compliance',
            'Item': 'Average Compliance Index',
            'Metric': f"{avg_comp:.2f}%"
        })
        stats_summary.append({
            'Category': 'General Compliance',
            'Item': 'Total Violations Detected',
            'Metric': len(violations)
        })
        
        df_stats = pd.DataFrame(stats_summary)

        # 5. Write to multi-sheet Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_incidents.to_excel(writer, sheet_name='Safety Incidents', index=False)
            df_compliance.to_excel(writer, sheet_name='Compliance History', index=False)
            df_stats.to_excel(writer, sheet_name='KPI Summary', index=False)
            
        logger.info(f"Excel report successfully created at: {file_path}")
        return file_path
