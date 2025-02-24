import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


def test():
    from model.model import get_model, predict
    from model.preprocessor import get_transform, preprocess_img

    classifier = get_model(
        num_classes=2,
    )
    classifier.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "../weights/mobilenet_v1.pth",
            ),
            weights_only=True,
        )
    )
    classifier.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_img = os.path.join(
        os.path.dirname(__file__),
        "../chair.jpg",
    )
    transform = get_transform(
        augmentation=True,
    )
    tensor_img = preprocess_img(
        predict_img,
        transform,
    )
    tensor_img = tensor_img.to(device)
    pred = predict(
        classifier,
        tensor_img,
        device=device,
        class_labels=["cat", "dog"],
    )
    print(pred)


test()
