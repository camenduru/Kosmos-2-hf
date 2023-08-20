import gradio as gr
import random
import numpy as np
import os
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2

colors = [
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),

    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),

    (255, 128, 0),
    (255, 0, 128),
    (0, 255, 128),

    (128, 255, 0),
    (128, 0, 255),
    (0, 128, 255),

    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 255),

    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),

    (255, 128, 64),
    (255, 64, 128),
    (64, 255, 128),

    (128, 255, 64),
    (128, 64, 255),
    (64, 128, 255),

    (255, 64, 64),
    (64, 255, 64),
    (64, 64, 255),

    (64, 255, 255),
    (255, 64, 255),
    (255, 255, 64),

    (128, 64, 64),
    (64, 128, 64),
    (64, 64, 128),

    (64, 128, 128),
    (128, 64, 128),
    (128, 128, 64),

    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),

    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),

    (64, 64, 0),
    (64, 0, 64),
    (0, 64, 64),

    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),

    (255, 64, 0),
    (255, 0, 64),
    (0, 255, 64),

    (64, 255, 0),
    (64, 0, 255),
    (0, 64, 255),

    (128, 64, 0),
    (128, 0, 64),
    (0, 128, 64),

    (64, 128, 0),
    (128, 0, 255),
    (0, 64, 128),
]

color_map = {f"color_id_{color_id}": "red" for color_id, color in enumerate(colors)}


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities, show=False, save_path=None):
    """_summary_
    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        # pdb.set_trace()
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 2
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_name, (start, end), bboxes in entities:
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            if start is None and bbox_id > 0:
                color_id += 1
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)

            # draw bbox
            # random color
            color = used_colors[bbox_id]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                    text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                    y1 += (text_height + text_offset_original + 2 * text_spaces)

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)
    if show:
        pil_image.show()

    return pil_image


def main():

    ckpt = "ydshieh/kosmos-2-patch14-224"

    model = AutoModelForVision2Seq.from_pretrained(ckpt, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    def generate_predictions(image_input, text_input, do_sample, sampling_topp, sampling_temperature):

        user_image_path = "/tmp/user_input_test_image.jpg"
        # This will be of `.jpg` format
        image_input.save(user_image_path)
        # This might give different results from the original argument `image_input`
        image_input = Image.open(user_image_path)

        if text_input == "Brief":
            text_input = "<grounding>An image of"
        elif text_input == "Detailed":
            text_input = "<grounding>Describe this image in detail:"
        else:
            text_input = f"<grounding>{text_input}"

        inputs = processor(text=text_input, images=image_input, return_tensors="pt")

        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"][:, :-1],
            attention_mask=inputs["attention_mask"][:, :-1],
            img_features=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # By default, the generated  text is cleanup and the entities are extracted.
        processed_text, entities = processor.post_processor_generation(generated_text)

        annotated_image = draw_entity_boxes_on_image(image_input, entities, show=True)

        color_id = -1
        entity_info = []
        for entity_name, (start, end), bboxes in entities:
            color_id += 1
            for bbox_id, _ in enumerate(bboxes):
                if start is None and bbox_id > 0:
                    color_id += 1
            if start is not None:
                entity_info.append(((start, end), color_id))

        colored_text = []
        prev_start = 0
        end = 0
        for idx, ((start, end), color_id) in enumerate(entity_info):
            if start > prev_start:
                colored_text.append((processed_text[prev_start:start], None))
            colored_text.append((processed_text[start:end], f"color_id_{color_id}"))
            prev_start = start

        if end < len(processed_text):
            colored_text.append((processed_text[end:len(processed_text)], None))

        return annotated_image, colored_text

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
                                ).style(color_map=color_map)

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

    demo.launch()


if __name__ == "__main__":
    main()
