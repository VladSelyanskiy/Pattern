# python
import xml.etree.ElementTree as ET

# 3rdparty
from PIL import Image

"""В строках помеченных * следует указать путь, следуя из названия переменной"""

path_to_annotation = "...\\annotation"  # *
tree = ET.parse(path_to_annotation)
root = tree.getroot()

coordinates = (
    []
)  # список который будет содержать название картинки и координаты вырезки
for elem in root:
    name_of_picture = elem.attrib.get("name", -1)  # название файла из аннотации
    if (
        name_of_picture != -1
    ):  # проверка на наличие аттрибута имени в элемента (имя есть только в нужных элементах)
        for subelem in elem:  # сохранение координат вырезки
            x_y = [
                subelem.attrib["xtl"],
                subelem.attrib["ytl"],
                subelem.attrib["xbr"],
                subelem.attrib["ybr"],
            ]
            x_y = tuple(
                map(lambda x: round(float(x)), x_y)
            )  # преобразование координат в нужный тип
            coordinates.append(
                (name_of_picture, x_y)
            )  # добавление в созданный ранее список

# для проверки координат
# print(coordinates)

c = 0  # счетчик для подсчета количества кропов
path_to_images = "...\\directory"  # *
path_of_savefile = "...\\directory"  # *
for element in coordinates:
    c += 1
    with Image.open(f"{path_to_images}{element[0]}") as im:  # октрытие картинки
        im_crop = im.crop(element[-1])  # вырезка по координатам
        im_crop.save(path_of_savefile)  # сохранение вырезанной картинки
