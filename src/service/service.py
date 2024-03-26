"""
Нейросетевой сервис FastAPI, 
который принимает спутниковый снимок и отдаёт наружу коллекцию объектов (JSON), 
причем в каждом для каждого из объектов хранятся координаты рамок и класс объекта


Детекция и классификация реализуется через нейросеть YOLOv8. 
Примерный код для использования resnet152 закомментирован, и пока не поддерживается.
(В сервисе выводится класс найденного объекта, но не всегда корректно)
"""

# python
import io
import json
import logging

# 3rdparty
# import cv2
import pydantic
import numpy as np

from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

app = FastAPI()

service_config_path = "service\\configs\\service_config.json"
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)


# датакласс конфига сервиса
class ServiceConfig(pydantic.BaseModel):
    name_of_classifier: str
    path_to_classifier: str
    name_of_detector: str
    path_to_detector: str


service_config_adapter = pydantic.TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(service_config_json)


# датакласс выхода сервиса
class ServiceOutput(pydantic.BaseModel):

    xtl: int
    ytl: int
    xbr: int
    ybr: int
    class_name: str


class Object(pydantic.BaseModel):
    objects: list[ServiceOutput]


# инициализация сетей
# class_names = {0: "aircraft", 1: "ship"}
# classifier = load_classifier()
# transform = torchvision.Compose([...])

detector = None
if service_config_python.name_of_detector.lower() == "yolov8":
    detector = YOLO(service_config_python.path_to_detector)
else:
    raise Exception()

logging.info(f"Загружен классификатор {service_config_python.name_of_classifier}")
logging.info(f"Файл весов классификатора: {service_config_python.path_to_classifier}")


# logger.info(f"Загружена конфигурация сервиса по пути: {service_config_path}")


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def health_check() -> str:
    """
    Точка доступа для проверки жизни сервиса

    Возрващает:
        HTTP Статус код (ОК)
    """
    return '{"Status" : "OK"}'


@app.post("/file/")
async def inference(image: UploadFile = File(...)) -> JSONResponse:
    """_summary_

    Args:
        image (UploadFile, optional): _description_. Defaults to File(...).

    Returns:
        JSONResponse: _description_
    """
    image_content = await image.read()
    cv_image = np.array(Image.open(io.BytesIO(image_content)))

    logger.info(f"Принята картинка размерности: {cv_image.shape}")

    # создаете объект выхода сервиса
    output_dict = {"objects": []}

    # выполнение детектора
    detector_outputs = detector(cv_image)

    class_names = {0: "aircraft", 1: "ship"}

    for result in detector_outputs:
        boxes = result.boxes
        for box in boxes:
            c = tuple(box.xyxy[0])
            xtl, ytl, xbr, ybr = int(c[0]), int(c[1]), int(c[2]), int(c[3])

            # crop_object = cv_image[ytl:ybr, xtl:xbr]
            # crop_tensor = transform(crop-object)
            # class_id = classifier.inference(crop_tensor)
            # class_name = class_names[class_id]

            class_name = str(class_names[box.cls[0].item()])

            output_dict["objects"].append(
                {
                    "xtl": xtl,
                    "xbr": xbr,
                    "ytl": ytl,
                    "ybr": ybr,
                    "class_name": class_name,
                }
            )

        for element in output_dict["objects"]:
            element = ServiceOutput(**element)

        service_output = Object(**output_dict)

        service_output_json = service_output.model_dump(mode="json")

        logging.info(f"Отправлена на сервис коллекция объектов")

        return JSONResponse(content=jsonable_encoder(service_output_json))


"""
Команды для запуска сервиса (указан относительный путь):

uvicorn service.service:app
uvicorn service.service:app --log-config=service\\log_config.yaml
"""
