import numpy as np
from abc import abstractmethod

from ..config import InpainterConfig
from ..utils import InfererModule, ModelWrapper

class CommonInpainter(InfererModule):

    async def inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        return await self._inpaint(image, mask, config, inpainting_size, verbose)

    @abstractmethod
    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass

    async def inpaint_batch(self, images: list, masks: list, config: InpainterConfig, inpainting_size: int = 1024) -> list:
        if hasattr(self, '_infer_batch'):
            return await self._infer_batch(images, masks, config, inpainting_size)
        return [await self._inpaint(img, mask, config, inpainting_size) for img, mask in zip(images, masks)]

class OfflineInpainter(CommonInpainter, ModelWrapper):
    _MODEL_SUB_DIR = 'inpainting'

    async def _inpaint(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass
