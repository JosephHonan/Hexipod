"""
Compatibility shim: Brax 0.14.x calls jax.device_put_replicated / device_put_sharded,
which newer JAX removed. These drop-in versions use jax.device_put + a sharding,
the replacement JAX's migration guide recommends. Import before training.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _sharding(devices):
    mesh = jax.sharding.Mesh(np.array(devices), ("devices",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))


def _stack(xs):
    return jnp.stack(xs) if isinstance(xs[0], jax.Array) else np.stack(xs)


def device_put_replicated(x, devices):
    s = _sharding(devices)
    return jax.tree.map(lambda a: jax.device_put(_stack([a] * len(devices)), s), x)


def device_put_sharded(shards, devices):
    s = _sharding(devices)
    return jax.tree.map(lambda *xs: jax.device_put(_stack(list(xs)), s), *shards)


for _name, _fn in (("device_put_replicated", device_put_replicated),
                   ("device_put_sharded", device_put_sharded)):
    try:
        getattr(jax, _name)
    except AttributeError:          # only patch when JAX no longer provides it
        setattr(jax, _name, _fn)
