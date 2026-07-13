from __future__ import annotations

import logging
import webbrowser
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    from vision_toolkit.stats.coco_dashboard import run_dashboard_export

    def run_analysis() -> None:
        json_path = filedialog.askopenfilename(
            title="Select COCO JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not json_path:
            messagebox.showinfo("Info", "No JSON file selected")
            return

        output_dir = filedialog.askdirectory(
            title="Choose output directory for analysis"
        )
        if not output_dir:
            output_dir = str(Path.home() / "coco_analysis")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(output_dir) / f"coco_analysis_{timestamp}")

        try:
            paths = run_dashboard_export(json_path, output_dir)
        except Exception as e:
            logger.exception("Analysis failed")
            messagebox.showerror("Error", f"Error during analysis: {e}")
            return

        try:
            webbrowser.open(f"file://{Path(paths['dashboard']).resolve()}")
        except OSError as e:
            logger.warning("Could not open browser: %s", e)

        messagebox.showinfo(
            "Analysis Complete",
            f"Results saved to: {output_dir}",
        )

    root = tk.Tk()
    root.title("COCO Annotation Analysis")
    root.geometry("420x220")

    tk.Label(root, text="COCO Annotation Analysis", font=("Arial", 16, "bold")).pack(
        pady=20
    )
    tk.Button(
        root,
        text="Start Analysis",
        command=run_analysis,
        font=("Arial", 12),
        padx=20,
        pady=10,
    ).pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gui()