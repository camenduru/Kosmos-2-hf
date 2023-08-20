import gradio as gr

def main():

    def generate_predictions(image_input, text_input, do_sample, sampling_topp, sampling_temperature):

        return None, None

    term_of_use = """
    ### Terms of use  
    By using this model, users are required to agree to the following terms:  
    The model is intended for academic and research purposes. 
    The utilization of the model to create unsuitable material is strictly forbidden and not endorsed by this work. 
    The accountability for any improper or unacceptable application of the model rests exclusively with the individuals who generated such content. 
    
    ### License
    This project is licensed under the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct).
    """

    with gr.Blocks(title="Kosmos-2", theme=gr.themes.Base()).queue() as demo:
        gr.Markdown(("""
            # Kosmos-2: Grounding Multimodal Large Language Models to the World
            [[Paper]](https://arxiv.org/abs/2306.14824) [[Code]](https://github.com/microsoft/unilm/blob/master/kosmos-2)
            """))
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Test Image")
                text_input = gr.Radio(["Brief", "Detailed"], label="Description Type", value="Brief")
                do_sample = gr.Checkbox(label="Enable Sampling", info="(Please enable it before adjusting sampling parameters below)", value=False)
                with gr.Accordion("Sampling parameters", open=False) as sampling_parameters:
                    sampling_topp = gr.Slider(minimum=0.1, maximum=1, step=0.01, value=0.9, label="Sampling: Top-P")
                    sampling_temperature = gr.Slider(minimum=0.1, maximum=1, step=0.01, value=0.7, label="Sampling: Temperature")

                run_button = gr.Button(label="Run", visible=True)

            with gr.Column():
                image_output = gr.Image(type="pil")
                text_output1 = gr.HighlightedText(
                                    label="Generated Description",
                                    combine_adjacent=False,
                                    show_legend=True,
                                ).style(color_map={"box": "red"})

        with gr.Row():
            with gr.Column():
                gr.Examples(examples=[
                            ["images/two_dogs.jpg", "Detailed", False],
                            ["images/snowman.png", "Brief", False],
                            ["images/man_ball.png", "Detailed", False],
                        ], inputs=[image_input, text_input, do_sample])
            with gr.Column():
                gr.Examples(examples=[
                            ["images/six_planes.png", "Brief", False],
                            ["images/quadrocopter.jpg", "Brief", False],
                            ["images/carnaby_street.jpg", "Brief", False],
                        ], inputs=[image_input, text_input, do_sample])
        gr.Markdown(term_of_use)

        run_button.click(fn=generate_predictions,
                         inputs=[image_input, text_input, do_sample, sampling_topp, sampling_temperature],
                         outputs=[image_output, text_output1],
                         show_progress=True, queue=True)

    demo.launch(share=True)


if __name__ == "__main__":
    main()
