from arxiv_paper.citation_graph_viewer import build_app


demo = build_app()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
