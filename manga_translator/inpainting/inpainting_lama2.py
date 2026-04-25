# manga_translator/inpainting/inpainting_lama2.py
"""
Custom LaMa inpainter that uses external LaMa Cleaner server.
Uses the already running LaMa Cleaner server on localhost:8080
Falls back to lama_large if server fails.
"""

import io
import numpy as np
from PIL import Image
import requests
from typing import List

from .common import CommonInpainter


class LaMa2Inpainter(CommonInpainter):
    """
    LaMa inpainter using external LaMa Cleaner server.
    Falls back to lama_large if server is unavailable or fails.
    """
    
    _LAMA_SERVER_URL = "http://localhost:8080/inpaint"
    _use_fallback = False
    _fallback_inpainter = None
    
    def __init__(self):
        super().__init__()
        self._check_server()
    
    def _check_server(self):
        """Check if LaMa Cleaner server is available."""
        try:
            response = requests.get("http://localhost:8080", timeout=3)
            self._use_fallback = False
            print("? LaMa2: Using external LaMa Cleaner server")
        except:
            self._use_fallback = True
            print("? LaMa2: Server not available, using lama_large fallback")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback inpainter."""
        if self._fallback_inpainter is None:
            try:
                from .inpainting_lama import LamaInpainter
                self._fallback_inpainter = LamaInpainter()
                print("? LaMa2: Fallback initialized")
            except Exception as e:
                print(f"? LaMa2: Failed to initialize fallback: {e}")
                raise
    
    async def _inpaint_server(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint using external LaMa Cleaner server.
        Raises exception if fails - will trigger fallback.
        """
        
        # Convert numpy arrays to PIL Images
        img_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        
        # Convert to JPEG bytes
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        mask_bytes = io.BytesIO()
        mask_pil.save(mask_bytes, format='JPEG', quality=95)
        mask_bytes.seek(0)
        
        # Prepare multipart form data
        files = {
            'image': ('image.jpg', img_bytes, 'image/jpeg'),
            'mask': ('mask.jpg', mask_bytes, 'image/jpeg')
        }
        
        data = {
            'ldmSteps': '25',
            'ldmSampler': 'ddim',
            'hdStrategy': 'Crop',
            'hdStrategyCropMargin': '196',
            'hdStrategyCropTrigerSize': '800',
            'hdStrategyResizeLimit': '2048',
            'controlnet_method': 'control_v11p_sd15_canny'
        }
        
        # Send request to LaMa Cleaner server
        response = requests.post(
            self._LAMA_SERVER_URL,
            files=files,
            data=data,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Server error: {response.status_code}")
        
        # Convert response to numpy array
        result_pil = Image.open(io.BytesIO(response.content))
        result_np = np.array(result_pil)
        
        # Ensure RGB format
        if len(result_np.shape) == 2:
            result_np = np.stack([result_np] * 3, axis=-1)
        elif result_np.shape[2] == 4:
            result_np = result_np[:, :, :3]
        
        return result_np
    
    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, inpaint_size: int = None) -> np.ndarray:
        """
        Main inpainting logic with automatic fallback.
        """
        
        # If already using fallback, go straight there
        if self._use_fallback:
            if self._fallback_inpainter is None:
                self._init_fallback()
            return await self._fallback_inpainter._inpaint(image, mask, inpaint_size)
        
        # Try server first
        try:
            return await self._inpaint_server(image, mask)
            
        except requests.exceptions.ConnectionError:
            print("? LaMa2: Connection error, switching to fallback")
            self._use_fallback = True
            self._init_fallback()
            return await self._fallback_inpainter._inpaint(image, mask, inpaint_size)
            
        except requests.exceptions.Timeout:
            print("? LaMa2: Server timeout, switching to fallback")
            self._use_fallback = True
            self._init_fallback()
            return await self._fallback_inpainter._inpaint(image, mask, inpaint_size)
            
        except Exception as e:
            print(f"? LaMa2: Server error ({e}), switching to fallback")
            self._use_fallback = True
            self._init_fallback()
            return await self._fallback_inpainter._inpaint(image, mask, inpaint_size)
    
    async def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        inpaint_size: int = None,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Main inpainting method with validation.
        """
        
        if verbose:
            mode = "fallback (lama_large)" if self._use_fallback else "server"
            print(f"LaMa2 inpainting ({mode}) - Image: {image.shape}, Mask: {mask.shape}")
        
        # Validate inputs
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be RGB (H, W, 3), got {image.shape}")
        
        if len(mask.shape) != 2:
            raise ValueError(f"Mask must be 2D (H, W), got {mask.shape}")
        
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch: {image.shape[:2]} vs {mask.shape}")
        
        # Check if mask has content
        if not np.any(mask > 128):
            if verbose:
                print("No inpainting needed - empty mask")
            return image
        
        # Perform inpainting (with automatic fallback)
        result = await self._inpaint(image, mask, inpaint_size)
        
        if verbose:
            print(f"Inpainting complete - Result: {result.shape}")
        
        return result
    
    def download(self):
        """
        Called during initialization.
        Checks server availability and downloads fallback model if needed.
        """
        self._check_server()
        
        # If using fallback, ensure it's downloaded
        if self._use_fallback:
            if self._fallback_inpainter is None:
                self._init_fallback()
            if hasattr(self._fallback_inpainter, 'download'):
                self._fallback_inpainter.download()