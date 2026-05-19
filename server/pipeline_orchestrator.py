import asyncio
import pickle
import logging
from typing import List

import aiohttp

from manga_translator import Config
from server.instance import executor_instances, ExecutorInstance
from server.translation_pool import translation_pool

logger = logging.getLogger('manga_translator')

# Maps job_id → ExecutorInstance that holds the pending Context for stage2
_pending_executor: dict = {}

# ── Batch collection settings ─────────────────────────────────────────────────
GPU_BATCH_SIZE = 3          # max images per GPU batch
BATCH_COLLECT_TIMEOUT = 1.0 # seconds to wait for a full batch before dispatching


async def _call_simple(instance: ExecutorInstance, method_name: str, nonce: str, **kwargs) -> object:
    """Call /simple_execute/{method_name} on a worker, return unpickled result."""
    url = f"http://{instance.ip}:{instance.port}/simple_execute/{method_name}"
    payload = pickle.dumps(kwargs)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            data=payload,
            headers={'X-Nonce': nonce, 'Content-Type': 'application/octet-stream'},
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Worker {method_name} failed ({resp.status}): {text}")
            raw = await resp.read()
            return pickle.loads(raw)


# ── Batch queue ───────────────────────────────────────────────────────────────

class _PipelineItem:
    __slots__ = ('image', 'config', 'nonce', 'future')

    def __init__(self, image, config: Config, nonce: str, future: asyncio.Future):
        self.image = image
        self.config = config
        self.nonce = nonce
        self.future = future


_batch_collect: List[_PipelineItem] = []
_batch_lock: asyncio.Lock = None
_batch_processor_started: bool = False


async def translate_with_pipeline(image, config: Config, nonce: str):
    """Submit image to the GPU batch queue and wait for the translated result."""
    global _batch_lock, _batch_processor_started

    if _batch_lock is None:
        _batch_lock = asyncio.Lock()
    if not _batch_processor_started:
        _batch_processor_started = True
        asyncio.create_task(_batch_processor_loop())

    future = asyncio.get_event_loop().create_future()
    item = _PipelineItem(image=image, config=config, nonce=nonce, future=future)
    async with _batch_lock:
        _batch_collect.append(item)

    return await future


async def _batch_processor_loop():
    """Collect queued images into GPU batches and process them."""
    while True:
        # Wait for at least one image
        while True:
            async with _batch_lock:
                if _batch_collect:
                    break
            await asyncio.sleep(0.05)

        # Collect up to GPU_BATCH_SIZE images within BATCH_COLLECT_TIMEOUT
        t_start = asyncio.get_event_loop().time()
        while True:
            async with _batch_lock:
                n = len(_batch_collect)
            if n >= GPU_BATCH_SIZE:
                break
            if asyncio.get_event_loop().time() - t_start >= BATCH_COLLECT_TIMEOUT:
                break
            await asyncio.sleep(0.05)

        async with _batch_lock:
            batch = _batch_collect[:GPU_BATCH_SIZE]
            del _batch_collect[:GPU_BATCH_SIZE]

        if batch:
            asyncio.create_task(_process_batch(batch))


async def _process_batch(batch: List[_PipelineItem]):
    """Process a batch: stage1_batch (sequential GPU) → translate (parallel API) → stage2_batch (GPU)."""
    nonce = batch[0].nonce
    logger.info(f"[batch-pipeline] Processing batch of {len(batch)} images")

    # ── Stage 1: one worker, all images sequentially ──────────────────────────
    instance: ExecutorInstance = await executor_instances.find_executor()
    try:
        stage1_results: list = await _call_simple(
            instance, 'translate_stage1_batch', nonce,
            images=[item.image for item in batch],
            config=batch[0].config,
        )
    except Exception as e:
        await executor_instances.free_executor(instance)
        for item in batch:
            if not item.future.done():
                item.future.set_exception(e)
        return

    # Separate valid jobs from empty/error results
    valid_pairs = []  # (item, s1_result)
    for item, s1 in zip(batch, stage1_results):
        if s1.get('error'):
            item.future.set_exception(RuntimeError(s1['error']))
        elif not s1.get('job_id'):
            from manga_translator.utils import Context
            ctx = Context()
            ctx.result = item.image
            item.future.set_result(ctx)
        else:
            valid_pairs.append((item, s1))

    if not valid_pairs:
        await executor_instances.free_executor(instance)
        return

    # Store worker reference for all pending jobs, then free worker during translation
    for _, s1 in valid_pairs:
        _pending_executor[s1['job_id']] = instance
    await executor_instances.free_executor(instance)

    # ── Translation: all in parallel ──────────────────────────────────────────
    translated_results = await asyncio.gather(
        *[translation_pool.translate(s1['texts'], s1['from_lang'], item.config)
          for item, s1 in valid_pairs],
        return_exceptions=True,
    )

    # Build stage2 job list
    jobs_for_stage2 = []
    items_for_stage2 = []
    for (item, s1), translated in zip(valid_pairs, translated_results):
        if isinstance(translated, Exception):
            _pending_executor.pop(s1['job_id'], None)
            item.future.set_exception(translated)
        else:
            jobs_for_stage2.append({'job_id': s1['job_id'], 'translated_texts': translated})
            items_for_stage2.append(item)

    if not jobs_for_stage2:
        return

    # ── Stage 2 batch: same worker (holds all pending contexts) ───────────────
    target_instance = _pending_executor.get(jobs_for_stage2[0]['job_id'])
    target_instance = await executor_instances.wait_for_specific_executor(target_instance)
    try:
        logger.info(f"[batch-pipeline] stage2_batch {len(jobs_for_stage2)} images on :{target_instance.port}")
        ctx_list: list = await _call_simple(
            target_instance, 'translate_stage2_batch', nonce,
            jobs=jobs_for_stage2,
            config=batch[0].config,
        )
    except Exception as e:
        logger.error(f"[batch-pipeline] stage2_batch failed: {e}")
        for item in items_for_stage2:
            if not item.future.done():
                item.future.set_exception(e)
        return
    finally:
        await executor_instances.free_executor(target_instance)
        for job in jobs_for_stage2:
            _pending_executor.pop(job['job_id'], None)

    # Resolve individual futures
    for item, ctx in zip(items_for_stage2, ctx_list):
        if not item.future.done():
            item.future.set_result(ctx)
