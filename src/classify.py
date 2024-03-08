# python
import os
import argparse
from typing import Any

# 3rdarty
import torch

import torchvision.transforms as transforms

import torch.nn as nn
import torchvision.models as models

from PIL import Image
import cv2


def inference_classifier(classifier: Any, path_to_image) -> str:
    """Метод для инференса классификатора на единичном изображении
    Args:
        classifier (Any): Модель, которая будет классифицировать объекты
        path_to_image (_type_): Путь к изображению

    Returns:
        str: Название классифицированного объекта
    """

    transform = transforms.Compose([transforms.ToTensor()])

    image = Image.open(path_to_image)
    image = transform(image)
    image = image.unsqueeze(0)

    classifier.eval()
    preds = classifier(image)
    pred = preds.argmax()

    labels = {0: "ship", 1: "plane"}

    return labels[int(pred)]


def load_classifier(
    name_of_classifier: str, path_to_pth_weights: str, device="cpu"
) -> Any:
    """Метод для загрузки класификатора
    Args:
        name_of_classifier (str): Название модели
        path_to_pth_weights (str): Путь к весам нжной модели
        device (str): На чем будут происходить вычисления

    Returns:
        Any: Модель(классификатор) с загруженными весами
    """

    possible_models = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }

    model = possible_models[name_of_classifier]()

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500, 2)
    )

    model.load_state_dict(torch.load(path_to_pth_weights))

    return model


def arguments_parser() -> argparse.Namespace:
    """
    Парсер аргументов
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения классификатора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_classifier", "-nc", type=str, help="Название классификатора"
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PTH-файлу с весами классификатора",
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке с изображениями",
    )
    parser.add_argument(
        "--use_cuda",
        "-uc",
        action="store_true",
        help="Использовать ли CUDA для инференса",
        default="",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Основная логика работы с классификатором"""
    args = arguments_parser()

    name_of_classifier = args.name_of_classifier
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    use_cuda = args.use_cuda

    print(f"Name of classifier: {name_of_classifier}")
    print(f"Path to content: {path_to_content}")
    print(f"Path to weights: {path_to_weights}")

    if use_cuda:
        print("Device: CUDA")
    else:
        print("Device: CPU")

    classifier = load_classifier(
        name_of_classifier=name_of_classifier, path_to_pth_weights=path_to_weights
    )

    if os.path.isfile(path_to_content):
        image = cv2.imread(path_to_content)

        image = cv2.resize(image, (280, 280))
        cv2.putText(
            image,
            f"{inference_classifier(classifier=classifier, path_to_image=path_to_content)}",
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("image", image)

        button = cv2.waitKey(0)
        if button == ord("q"):
            cv2.destroyAllWindows()
    else:
        data = os.listdir(path_to_content)
        number = 0

        while True:
            path = path_to_content + "\\" + data[number]
            image = cv2.imread(path)
            image = cv2.resize(image, (280, 280))
            cv2.putText(
                image,
                f"{inference_classifier(classifier=classifier, path_to_image=path)}",
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.imshow("image", image)

            button = chr(cv2.waitKey(0))

            if button == "d":
                number = number + 1
                if number > len(data) - 1:
                    number = 0
            if button == "a":
                number = number - 1
                if number < 0:
                    number = len(data) - 1
            if button == "q":
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
