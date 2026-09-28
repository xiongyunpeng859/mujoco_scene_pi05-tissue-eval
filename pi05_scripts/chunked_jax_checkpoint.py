"""Single-device streaming Orbax handler, with exact per-block readback.

Uses the installed Orbax internal API; must pass roundtrip tests after upgrades.
No dtype conversion, lossy compression, or changes to the parameter tree.
"""
import dataclasses
import math
import jax
import numpy as np
import tensorstore as ts
from orbax.checkpoint._src.serialization import type_handlers as handlers


class ChunkedArrayHandler(handlers.ArrayHandler):
    def __init__(self, block_bytes=64 * 1024**2):
        super().__init__()
        self.block_bytes = block_bytes

    async def serialize(self, values, infos, args=None):
        args = args or [handlers.SaveArgs()] * len(values)
        self._ext_metadata = {}
        metadata=[]
        transaction=ts.Transaction()
        for value, info, arg in zip(values, infos, args):
            if jax.process_count()!=1 or len(value.devices())!=1:
                raise ValueError('Streaming save currently supports one process and one device only')
            shape=tuple(value.shape)
            if not all(shape): raise ValueError('Empty tensors are not supported')
            axis=max(range(len(shape)), key=lambda i: shape[i]) if shape else 0
            plane=math.prod(shape[:axis]+shape[axis+1:])*value.dtype.itemsize
            if plane > self.block_bytes:
                raise ValueError(f'One tensor slice exceeds block budget: {info.name}')
            width=max(1,self.block_bytes//plane)
            local=list(shape)
            if shape: local[axis]=min(shape[axis],width)
            spec=handlers._build_array_write_spec(
                info=info,arg=dataclasses.replace(arg,chunk_byte_size=self.block_bytes),
                global_shape=shape,local_shape=tuple(local),dtype=value.dtype,
                use_ocdbt=info.is_ocdbt_checkpoint,
                process_index=handlers.get_process_index_for_subdir(info.is_ocdbt_checkpoint),
                metadata_key=self._metadata_key)
            store=await ts.open(spec.json,create=True,open=True,context=info.ts_context)
            for start in range(0,shape[axis] if shape else 1,width):
                index=[slice(None)]*len(shape)
                if shape: index[axis]=slice(start,min(start+width,shape[axis]))
                index=tuple(index)
                # Transfer a slice, not a whole parameter or the entire model.
                host=np.asarray(jax.device_get(value[index] if shape else value))
                await store[index].write(host)
                restored=await store[index].read()
                if restored.shape!=host.shape or restored.dtype!=host.dtype or restored.tobytes()!=host.tobytes():
                    raise IOError(f'Checkpoint block verification failed: {info.name} {index}')
                del restored,host
            if self._enable_write_sharding_file:
                await self._serialize_sharding(value.sharding, info, transaction)
            metadata.append(spec.metadata)
            print(f'CHECKPOINT_STREAM_VERIFIED {info.name} shape={shape}',flush=True)
        if self._array_metadata_store is not None:
            await self._array_metadata_store.write(checkpoint_dir=infos[0].parent_dir,
                array_metadatas=metadata,process_index=0)
        await transaction.commit_async()
        return []


def make_handler(block_bytes=64*1024**2):
    import orbax.checkpoint as ocp
    registry=handlers.create_type_handler_registry((jax.Array,ChunkedArrayHandler(block_bytes)))
    return ocp.PyTreeCheckpointHandler(type_handler_registry=registry)
