from IPython.display import Image, display

def visualize_workflow(workflow):
    display(Image(workflow.get_graph().draw_mermaid_png())) 