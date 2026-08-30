# -*- coding: utf-8 -*-

from mobileo.model.qwen25vl_wrapper import Qwen25VLWrapper


MODEL_PATH = (
    r"E:\ai for science\omnifuse\models"
    r"\Qwen\Qwen2___5-VL-3B-Instruct"
)

VISIBLE_PATH = r"E:\ai for science\SeAFusion-main\MSRS\Visible\train\MSRS\00001D.png"
INFRARED_PATH = r"E:\ai for science\SeAFusion-main\MSRS\Infrared\train\MSRS\00001D.png"


def main():

    qwen = Qwen25VLWrapper(
        model_path=MODEL_PATH,
        last_n_layers=8,
        freeze=True,
        device="cuda",
    )

    qwen.print_hidden_shapes(
        visible=VISIBLE_PATH,
        infrared=INFRARED_PATH,
    )


if __name__ == "__main__":

    main()