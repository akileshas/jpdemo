import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


def test():
    from model.dataset import get_dataloader

    train_loader = get_dataloader()

    for data, label in train_loader:
        print(data.shape)
        print(label)
        break

    all_labels = []
    for _, label in train_loader:
        all_labels.extend(label.numpy().tolist())

    print("Unique Labels:", set(all_labels))


test()
