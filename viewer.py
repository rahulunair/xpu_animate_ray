import gradio as gr
from pathlib import Path
import re

def list_animations():
    """List all animations from the generated_animations directory."""
    path = Path("generated_animations")
    if not path.exists():
        return [], []

    files = []
    prompts = []
    for file in sorted(path.glob("animation_*.gif")):
        match = re.match(r'animation_\d{8}_\d{6}_(.+)\.gif', file.name)
        if match:
            files.append(str(file))
            prompts.append(match.group(1).replace('_', ' '))

    return files, prompts

def update_gallery(search=""):
    files, prompts = list_animations()
    if search:
        filtered = [(f, p) for f, p in zip(files, prompts) if search.lower() in p.lower()]
        files = [f[0] for f in filtered]
        prompts = [f[1] for f in filtered]
    return files

with gr.Blocks() as demo:
    gr.Markdown("# AnimateDiff Generated Animations Gallery")

    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh")
        search = gr.Textbox(label="Search prompts", placeholder="Enter search terms...")

    gallery = gr.Gallery(
        value=list_animations()[0],
        columns=3,
        height=600,
        show_label=False
    )

    refresh_btn.click(
        fn=update_gallery,
        inputs=[search],
        outputs=gallery
    )

    search.change(
        fn=update_gallery,
        inputs=[search],
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
