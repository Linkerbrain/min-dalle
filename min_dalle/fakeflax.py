import msgpack
import enum
import numpy as np
import struct

"""
FLAX DOES NOT WORK ON WINDOWS SO I HACKED TOGETHER THE FUNCTIONS THAT WERE NEEDED

TO BE FIXED LATER
"""

# the empty node is a struct.dataclass to be compatible with JAX.
# @struct.dataclass
class _EmptyNode:
  pass

empty_node = _EmptyNode()


_dict_to_tuple = lambda dct: tuple(dct[str(i)] for i in range(len(dct)))

class _MsgpackExtType(enum.IntEnum):
  """Messagepack custom type ids."""
  ndarray = 1
  native_complex = 2
  npscalar = 3

def _dtype_from_name(name: str):
  """Handle JAX bfloat16 dtype correctly."""
  if name == b'bfloat16':
    raise NotImplementedError() # return jax.numpy.bfloat16
  else:
    return np.dtype(name)

def _unchunk(data):
  """Convert canonical dictionary of chunked arrays back into array."""
  assert '__msgpack_chunked_array__' in data
  shape = _dict_to_tuple(data['shape'])
  flatarr = np.concatenate(_dict_to_tuple(data['chunks']))
  return flatarr.reshape(shape)

def _ndarray_from_bytes(data: bytes) -> np.ndarray:
  """Load ndarray from simple msgpack encoding."""
  shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
  return np.frombuffer(buffer,
                       dtype=_dtype_from_name(dtype_name),
                       count=-1,
                       offset=0).reshape(shape, order='C')

def _msgpack_ext_unpack(code, data):
  """Messagepack decoders for custom types."""
  if code == _MsgpackExtType.ndarray:
    return _ndarray_from_bytes(data)

  elif code == _MsgpackExtType.native_complex:
    complex_tuple = msgpack.unpackb(data)
    return complex(complex_tuple[0], complex_tuple[1])

  elif code == _MsgpackExtType.npscalar:
    ar = _ndarray_from_bytes(data)

    return ar[()]  # unpack ndarray to scalar
  return msgpack.ExtType(code, data)

def _unchunk_array_leaves_in_place(d):
  """Convert chunked array leaves back into array leaves, in place."""
  if isinstance(d, dict):
    if '__msgpack_chunked_array__' in d:
      return _unchunk(d)
    else:
      for k, v in d.items():
        if isinstance(v, dict) and '__msgpack_chunked_array__' in v:
          d[k] = _unchunk(v)
        elif isinstance(v, dict):
          _unchunk_array_leaves_in_place(v)
  return d

def msgpack_restore(encoded_pytree: bytes):
  """Restore data structure from bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `from_bytes`.

  Args:
    encoded_pytree: msgpack-encoded bytes of python tree.

  Returns:
    Python tree of dict, list, tuple with python primitive
    and array leaves.
  """
  state_dict = msgpack.unpackb(
      encoded_pytree, ext_hook=_msgpack_ext_unpack, raw=False)
  return _unchunk_array_leaves_in_place(state_dict)

def flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
  """Flatten a nested dictionary.

  The nested keys are flattened to a tuple.
  See `unflatten_dict` on how to restore the
  nested dictionary structure.

  Example::

    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_xs = flatten_dict(xs)
    print(flat_xs)
    # {
    #   ('foo',): 1,
    #   ('bar', 'a'): 2,
    # }

  Note that empty dictionaries are ignored and
  will not be restored by `unflatten_dict`.

  Args:
    xs: a nested dictionary
    keep_empty_nodes: replaces empty dictionaries
      with `traverse_util.empty_node`. This must
      be set to `True` for `unflatten_dict` to
      correctly restore empty dictionaries.
    is_leaf: an optional function that takes the
      next nested dictionary and nested keys and
      returns True if the nested dictionary is a
      leaf (i.e., should not be flattened further).
    sep: if specified, then the keys of the returned
      dictionary will be `sep`-joined strings (if
      `None`, then keys will be tuples).
  Returns:
    The flattened dictionary.
  """
  def _key(path):
    if sep is None:
      return path
    return sep.join(path)

  def _flatten(xs, prefix):
    if not isinstance(xs, (dict)) or (
        is_leaf and is_leaf(prefix, xs)):
      return {_key(prefix): xs}
    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      path = prefix + (key,)
      result.update(_flatten(value, path))
    if keep_empty_nodes and is_empty:
      return {_key(prefix): empty_node}
    return result
  return _flatten(xs, ())