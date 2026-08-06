import win32com.client
import os

pptx_path = r"C:\Users\Vishakha.Roy\Downloads\Deepfake\Group-11-DS-and-AI-Lab-Project\doc\Milestone-5\ppt\Milestone5_Presentation.pptx"
out_dir = r"C:\Users\Vishakha.Roy\Downloads\Deepfake\Group-11-DS-and-AI-Lab-Project\doc\Milestone-5\ppt\slides_png"
os.makedirs(out_dir, exist_ok=True)

powerpoint = win32com.client.Dispatch("PowerPoint.Application")
powerpoint.Visible = 1
pres = powerpoint.Presentations.Open(pptx_path, WithWindow=False)
pres.Export(out_dir, "PNG", 1280, 720)
pres.Close()
powerpoint.Quit()
print("Exported to", out_dir)
