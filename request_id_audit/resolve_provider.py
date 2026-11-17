"""Resolve the original function in TF's locally loaded ELF libraries."""
import ctypes
import os
from pathlib import Path

SYMBOL = '_ZN10tensorflow18GetUniqueRequestIdEv'


def register_provider(tf):
    library = os.environ.get('FFN_REQUEST_ID_LIBRARY')
    if not library:
        return
    hook = ctypes.CDLL(library)
    hook_address = ctypes.cast(getattr(hook, SYMBOL), ctypes.c_void_p).value
    setter = hook.ffn_id_set_original
    setter.argtypes = [ctypes.c_void_p]
    setter.restype = None
    root = Path(tf.sysconfig.get_lib())
    candidates = [root / 'libtensorflow_cc.so.2',
                  root / 'libtensorflow_framework.so.2',
                  root / 'python' / '_pywrap_tensorflow_internal.so']
    # Inspect only this process's already loaded TF modules. RTLD_NOLOAD below
    # prevents loading additional TensorFlow libraries or running new initializers.
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and 'tensorflow' in fields[5] and '.so' in fields[5]:
            candidates.append(Path(fields[5]))
    checked = set()
    failures = []
    for path in candidates:
        if str(path) in checked:
            continue
        checked.add(str(path))
        try:
            provider = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
            native = getattr(provider, SYMBOL)
            address = ctypes.cast(native, ctypes.c_void_p).value
        except (OSError, AttributeError) as e:
            failures.append(str(e))
            continue
        if address and address != hook_address:
            setter(address)
            # Retain the explicit library handle for the lifetime of the process.
            register_provider.provider = provider
            print('FFN_ID_PROVIDER_REGISTERED', str(path),
                  'native=', hex(address), 'hook=', hex(hook_address), flush=True)
            return
    raise RuntimeError('Cannot resolve native GetUniqueRequestId independently '
                       'of the hook. No training started. Details: ' + '; '.join(failures))
