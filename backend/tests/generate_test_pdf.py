from fpdf import FPDF
import os

pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

# Paragraph
pdf.multi_cell(0, 10, "This is a test document to evaluate table extraction. Below is a table containing performance metrics of different models.")
pdf.ln(5)

# Table
data = (
    ("Model Name", "Latency (ms)", "Accuracy (%)", "VRAM (GB)"),
    ("Llama-3.2-1B", "150", "85.5", "2.0"),
    ("Qwen-2.5-1.5B", "180", "87.2", "2.5"),
    ("BGE-M3", "50", "92.0", "1.2"),
)

with pdf.table() as table:
    for data_row in data:
        row = table.row()
        for datum in data_row:
            row.cell(datum)

pdf.ln(5)
pdf.multi_cell(0, 10, "End of the document. This text should not be mixed up inside the table cells.")

output_path = os.path.join(os.path.dirname(__file__), "test_table.pdf")
pdf.output(output_path)
print(f"Generated test PDF with table at: {output_path}")
