import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


def test():
    import model.preprocessor as procs

    img_path = os.path.join(
        os.path.dirname(__file__),
        "../dataset/class-0/-Haaot_Sn4A_jpg.rf.54dcd119b808c18059b9f49b2d34b9c5.jpg",
    )

    transform = procs.get_transform(
        augmentation=True,
    )

    tensor_img = procs.preprocess_img(
        img_path=img_path,
        transforms=transform,
    )

    print(tensor_img.shape)


test()
