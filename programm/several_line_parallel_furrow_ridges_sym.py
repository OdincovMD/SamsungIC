import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, models
from typing import Optional, Callable, Any, List
import torch.nn as nn
import os
from skimage.io import imsave, imread
from skimage.transform import resize
import cv2

os.chdir(os.path.dirname(__file__))


class BlockBuilder:

    @staticmethod
    def create_enc_dec_block(in_dim: int, out_dim: int, is_last=False) -> nn.Sequential:
        block = []
        block.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_dim),
                nn.ReLU()
            )
        )
        if is_last:
            block.append(
                nn.Conv2d(in_channels=out_dim, out_channels=1,
                          kernel_size=3, padding=1)
            )
        else:
            block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=out_dim,
                              out_channels=out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=out_dim),
                    nn.ReLU()
                )
            )

        return nn.Sequential(*block)

    @staticmethod
    def create_pool_block(is_unpool=False, return_indices=True):
        if is_unpool:
            block = nn.MaxUnpool2d(2, stride=2)
        else:
            block = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        return block


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        builder = BlockBuilder()

        self.enc_conv0 = builder.create_enc_dec_block(3, 64)
        self.pool0 = builder.create_pool_block(return_indices=False)
        self.enc_conv1 = builder.create_enc_dec_block(64, 128)
        self.pool1 = builder.create_pool_block(return_indices=False)
        self.enc_conv2 = builder.create_enc_dec_block(128, 256)
        self.pool2 = builder.create_pool_block(return_indices=False)
        self.enc_conv3 = builder.create_enc_dec_block(256, 512)
        self.pool3 = builder.create_pool_block(return_indices=False)

        self.bottleneck_conv = builder.create_enc_dec_block(512, 1024)

        self.upsample0 = nn.Upsample(32)
        self.dec_conv0 = builder.create_enc_dec_block(1024+512, 512)
        self.upsample1 = nn.Upsample(64)
        self.dec_conv1 = builder.create_enc_dec_block(512+256, 256)
        self.upsample2 = nn.Upsample(128)
        self.dec_conv2 = builder.create_enc_dec_block(256+128, 128)
        self.upsample3 = nn.Upsample(256)
        self.dec_conv3 = builder.create_enc_dec_block(128+64, 32, True)

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # decoder
        d0 = self.upsample0(b)
        d0 = torch.cat([d0, e3], dim=1)
        d0 = self.dec_conv0(d0)

        d1 = self.upsample1(d0)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec_conv1(d1)

        d2 = self.upsample2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec_conv2(d2)

        d3 = self.upsample3(d2)
        d3 = torch.cat([d3, e0], dim=1)
        d3 = self.dec_conv3(d3)  # no activation
        return d3


class CustomNeuralNetResNet(torch.nn.Module):
    def __init__(self, outputs_number):
        super(CustomNeuralNetResNet, self).__init__()
        self.net = models.resnet152(pretrained=True)

        # Выключаем переобучение весов каждого слоя модели, кроме последнего
        for param in self.net.parameters():
            param.requires_grad = False

        TransferModelOutputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, outputs_number)
        )

    def forward(self, x):
        return self.net(x)


class NumpyImageDataset(Dataset):
    """
    Датасет для работы с изображениями в формате массива NumPy.

    Атрибуты:
        image_array (np.ndarray): Массив изображений.
        transform (callable): Преобразование данных, которое применяется к изображению.

    Методы:
        __init__(self, image_array, transform=None): Конструктор класса.
        __len__(self): Возвращает количество элементов в датасете.
        __getitem__(self, idx): Возвращает элемент датасета по индексу.
    """

    def __init__(self, image_array: np.ndarray, transform: Optional[Callable[[Any], Any]] = None) -> None:
        self.image_array = image_array
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_array)

    def __getitem__(self, idx: int) -> Any:
        image = self.image_array[idx]
        if self.transform:
            image = self.transform(image)
        return image


def main(img: np.ndarray) -> str:
    """
    Основная функция для классификации изображения на симметрию и асимметрию.

    Args:
        img (np.ndarray): Массив изображения в формате numpy.

    Returns:
        str: Результат классификации: "Симметрия" или "Асимметрия".
    """

    # Создание датасета из массива изображения
    image_array_list = [img]

    # Трансформация изображения
    def transform(image) -> np.ndarray:
        image = resize(image, (256, 256), anti_aliasing=True)
        image = np.rollaxis(image, 2, 0)  # Перекладываем каналы
        return image

    # Создание датасета из массива изображения
    numpy_image_dataset = NumpyImageDataset(
        image_array_list, transform=transform)
    dataloader = DataLoader(numpy_image_dataset, batch_size=1, shuffle=False)

    # Загрузка весов в модель
    unet_model = UNet().to('cpu')
    unet_model.load_state_dict(torch.load(r'sg_several_line_parallel_furrow_ridges_sym.pth',
                                          map_location=torch.device('cpu')))

    unet_model.eval()

    # Маска новообразования
    save_dir = 'predictions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for inputs in dataloader:
            images = inputs.to('cpu', dtype=torch.float)
            outputs = unet_model(images)
            preds = torch.sigmoid(outputs)
            preds = preds > 0.7

            preds = preds.cpu().numpy()
            for _, pred in enumerate(preds):
                filename = os.path.join(save_dir, 'prediction.png')
                imsave(filename, pred[0])  # Сохраняем первую маску (бинарная)

    # Наложение маски
    mask = cv2.imread(r'predictions\prediction.png', cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (256, 256))
    img_resized = cv2.resize(img, (256, 256))
    masked_img = cv2.bitwise_and(img_resized, img_resized, mask=mask_resized)

    # Трансформации изображения
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Преобразование в объект PIL.Image
        transforms.Resize((224, 224)),
        transforms.CenterCrop(200),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Создание датасета из массива изображения
    image_array_list = [masked_img]
    numpy_image_dataset = NumpyImageDataset(
        image_array_list, transform=transform)
    dataloader = DataLoader(numpy_image_dataset, batch_size=1, shuffle=False)

    # Описание классов
    info = ["Симметрия", "Асимметрия"]

    # Загрузка модели и выполнение инференса
    some_line_model = CustomNeuralNetResNet(2)
    some_line_model.load_state_dict(torch.load(
        r'cl_several_line_parallel_furrow_ridges_sym.pth', map_location=torch.device('cpu')))
    some_line_model.eval()
    test_predictions = []
    for inputs in dataloader:
        with torch.set_grad_enabled(False):
            preds = some_line_model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
    pred = 0
    if test_predictions[0] < 0.45:
        pred = 1
    return info[pred]


if __name__ == "__main__":
    # type(img) is numpy.ndarray
    img = imread(r"c:\Users\hardb\Desktop\dataset\test\1_new_615.jpg")
    result = main(img)
