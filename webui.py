
import json
import gradio as gr
from common_table import main as table_rec

css = """
    #submit {
            margin: auto !important; 
            background: linear-gradient(to right, #ffe2c0, #fdc589); 
            color: #ea580c
            }
"""


def clear_image():
    return None, None, None, None

def image_to_excel(image):
    output_file = output_image = res = None
    try:
        result = table_rec(image)
        res = json.dumps(result, indent = 4, ensure_ascii=False)
        output_file = r'test/result.xls'
        output_image = r'test/result_ceil.png'
    except Exception as e:
        result = []
        res = str(e)
        output_file = None
        output_image = None
    finally:
        if len(result) == 0:
            output_file = output_image = None
        return output_file, output_image, res


def main():
    with gr.Blocks(title = "表格识别", css=css) as demo:
        with gr.Row():
            header = gr.Markdown("""
                        <div style="text-align: center; font-size: 25px; font-weight: bold;">
                        表格识别
                        </div>
                        """)
        with gr.Row():
            with gr.Column():
                upload_image = gr.Image(type='filepath', label = 'Upload Image')
                with gr.Row():
                    btn1 = gr.Button("清除")
                    btn2 = gr.Button("提交", elem_id="submit")
                with gr.Row():
                    examples = gr.Examples(examples = [r"test/1/1.jpg", r"test/1/2.jpg", r"test/1/3.jpg"],
                                           inputs = upload_image)

            with gr.Column():
                output_file = gr.File(label = "Download Excel")
                output_image = gr.Image(type='numpy', label = 'Output Image')
                text = gr.Text(label='Json Response')

        btn1.click(clear_image, outputs = [upload_image, output_file, output_image, text])
        btn2.click(image_to_excel, inputs = upload_image, outputs = [output_file, output_image, text])

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8000, share=False, inbrowser=False)


if __name__ == "__main__":
    main()