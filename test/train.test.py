import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


def test():
    import model.dataset as dataset
    import model.model as model
    import model.train as train

    # Load the dataset
    dataloader = dataset.get_dataloader()

    # Load the model
    classifier = model.get_model(num_classes=2)

    # train the model
    train.train_model(
        classifier,
        dataloader,
        num_epochs=10,
    )

    torch.save(classifier.state_dict(), "../weights/mobilenet_v1.pth")


test()
