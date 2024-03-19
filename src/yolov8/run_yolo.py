from ultralytics import YOLO
import argparse
import cv2


def arguments_parser() -> argparse.Namespace:
    """
    Парсер аргументов
    """
    parser = argparse.ArgumentParser(description="Скрипт для выполнения изображении")

    parser.add_argument(
        "--path_to_weights",
        "-w",
        type=str,
        help="Путь к PT-файлу с весами",
    )

    parser.add_argument(
        "--path_to_content",
        "-p",
        type=str,
        help="Путь к одиночному изображению",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = arguments_parser()

    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content

    print(f"Path to content: {path_to_content}")
    print(f"Path to weights: {path_to_weights}")

    model = YOLO(path_to_weights)

    results = model(path_to_content)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            img = cv2.imread(path_to_content)
            c = tuple(box.xyxy[0])
            img = cv2.rectangle(
                img,
                pt1=(int(c[0]), int(c[1])),
                pt2=(int(c[2]), int(c[3])),
                color=(255, 255, 255),
                thickness=3,
            )

            # left=int(c[0])
            # bottom=int(c[1])
            # right=int(c[2])
            # top=int(c[3])

    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
