from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._embed_only = nnx_utils.module_jit(model.embed_only)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        # TODO: del at cleanup
        # print(f'infer 1 {[(k, v.shape, v.dtype, type(v)) for k, v in outputs.items()]}')
        # infer 1 [('state', (1, 8), dtype('float32'), <class 'jaxlib.xla_extension.ArrayImpl'>), ('actions', (1, 256), dtype('float32'), <class 'jaxlib.xla_extension.ArrayImpl'>)]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # TODO: del at cleanup
        # print(f'infer 2 {[(k, v.shape, v.dtype, type(v)) for k, v in outputs.items()]}')
        # infer 2 [('actions', (256,), dtype('float32'), <class 'numpy.ndarray'>), ('state', (8,), dtype('float32'), <class 'numpy.ndarray'>)]
        return self._output_transform(outputs)
    
    @override
    def embed_only(self, obs: dict) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        outputs = self._embed_only(_model.Observation.from_dict(inputs))
        # TODO: del at cleanup
        # print(f'embed 1 {[(k, v.shape, v.dtype, type(v)) for k, v in outputs.items()]}')
        # embed 1 [('base_0_rgb', (1, 256, 2048), dtype(bfloat16), <class 'jaxlib.xla_extension.ArrayImpl'>), ('base_1_rgb', (1, 256, 2048), dtype(bfloat16), <class 'jaxlib.xla_extension.ArrayImpl'>), ('wrist_0_rgb', (1, 256, 2048), dtype(bfloat16), <class 'jaxlib.xla_extension.ArrayImpl'>)]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # TODO: del at cleanup
        # print(f'embed 2 {[(k, v.shape, v.dtype, type(v)) for k, v in outputs.items()]}')
        # embed 2 [('base_0_rgb', (256, 2048), dtype(bfloat16), <class 'numpy.ndarray'>), ('base_1_rgb', (256, 2048), dtype(bfloat16), <class 'numpy.ndarray'>), ('wrist_0_rgb', (256, 2048), dtype(bfloat16), <class 'numpy.ndarray'>)]
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
