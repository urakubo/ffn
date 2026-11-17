#!/usr/bin/env python3
"""FFN asynchronous PS training with the tested request-ID counter launcher.

Save this file as repo/distributed_train_idfixed.py.
Run: python repo/distributed_train_idfixed.py --job_name=ps ...
JSON launch: --config ffn_config.json --role server|worker0|worker1|...
Use --dry-run with --config to inspect settings without importing TensorFlow.
Legacy command-line flags remain supported; training uses train.py.
The existing request_id_audit/libffn_id_audit.so and resolve_provider.py are
still required. FFN_IDFIX_ASSET_DIR may specify their directory explicitly.
Stop all workers and PS before changing launchers. One PS is supported.
Do not restart individual processes in a live cluster: ID namespaces are reused.
"""
import argparse
import json
import re
import os
from pathlib import Path
import subprocess
import sys
import time


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _environment(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + ' must be an object')
    reserved = {'LD_PRELOAD', 'FFN_ID_MODE', 'FFN_ID_NAMESPACE',
                'FFN_REQUEST_ID_LIBRARY', 'FFN_ID_AUDIT_DIR'}
    result = {}
    for key, item in value.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or key in reserved:
            raise ValueError('invalid or launcher-managed environment key: ' + key)
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise ValueError(label + '.' + key + ' must be a string or integer')
        text = str(item)
        if '\0' in text:
            raise ValueError('NUL character in environment value: ' + key)
        result[key] = text
    return result


def _option_argument(key, value):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', key):
        raise ValueError('invalid option name: ' + key)
    if value is None:
        return None
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'))
    elif isinstance(value, bool):
        text = 'true' if value else 'false'
    elif isinstance(value, list):
        if not value or any(not isinstance(x, (str, int, float)) or
                            isinstance(x, bool) or ',' in str(x) for x in value):
            raise ValueError('list option must contain nonempty scalar entries: ' + key)
        text = ','.join(str(x) for x in value)
    elif isinstance(value, (str, int, float)):
        text = str(value)
    else:
        raise ValueError('unsupported option value: ' + key)
    if '\0' in text:
        raise ValueError('NUL character in option: ' + key)
    return '--' + key + '=' + text


def config_launch(argv):
    """Convert one JSON + role into ordinary FFN flags and pre-import env."""
    has_config = any(x == '--config' or x.startswith('--config=') for x in argv)
    if not has_config:
        if any(x in ('--role', '--dry-run') or x.startswith('--role=') for x in argv):
            raise ValueError('--role and --dry-run require --config')
        return argv, {}, False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--dry-run', action='store_true')
    selected = parser.parse_args(argv)
    try:
        with selected.config.open(encoding='utf-8-sig') as stream:
            config = json.load(stream, object_pairs_hook=_unique_json_object,
                               parse_constant=lambda x: (_ for _ in ()).throw(
                                   ValueError('non-finite JSON number: ' + x)))
    except OSError as e:
        raise ValueError('cannot read configuration: ' + str(e)) from e
    if not isinstance(config, dict):
        raise ValueError('JSON root must be an object')
    options = config.get('options')
    processes = config.get('processes')
    if not isinstance(options, dict) or not isinstance(processes, dict):
        raise ValueError('options and processes must be JSON objects')
    if set(options) & {'job_name', 'task_index', 'master', 'task', 'ps_tasks'}:
        raise ValueError('job_name/task_index/master/task/ps_tasks are derived from --role')
    hosts = options.get('worker_hosts')
    if isinstance(hosts, str):
        hosts = hosts.split(',')
    if not isinstance(hosts, list) or not hosts or any(
            not isinstance(h, str) or not h or ',' in h for h in hosts):
        raise ValueError('worker_hosts must be a nonempty list of addresses')
    expected = {'server'} | {'worker%d' % i for i in range(len(hosts))}
    if set(processes) != expected:
        raise ValueError('processes must contain exactly: ' + ', '.join(sorted(expected)))
    if selected.role not in expected:
        raise ValueError('unknown --role: ' + selected.role)
    common_env = _environment(config.get('environment', {}), 'environment')
    environments = {}
    for role, settings in processes.items():
        if not isinstance(settings, dict) or set(settings) - {'environment'}:
            raise ValueError('process ' + role + ' accepts environment only')
        env = {**common_env, **_environment(settings.get('environment', {}), role)}
        if role == 'server':
            if env.get('CUDA_VISIBLE_DEVICES', ''):
                raise ValueError('server must have CUDA_VISIBLE_DEVICES=""')
            env['CUDA_VISIBLE_DEVICES'] = ''
        elif not env.get('CUDA_VISIBLE_DEVICES') or ',' in env['CUDA_VISIBLE_DEVICES']:
            raise ValueError(role + ' must select one GPU with CUDA_VISIBLE_DEVICES')
        environments[role] = env
    arguments = [arg for key, value in options.items()
                 if key != 'tracker_sync_steps'
                 if (arg := _option_argument(key, value)) is not None]
    is_server = selected.role == 'server'
    arguments.extend(['--job_name=' + ('ps' if is_server else 'worker'),
                      '--task_index=' + ('0' if is_server else selected.role[6:])])
    namespace_for(arguments)
    return arguments, environments[selected.role], selected.dry_run


def flag_value(args, name, default=None):
    values = []
    key = '--' + name
    for i, arg in enumerate(args):
        if arg.startswith(key + '='):
            values.append(arg.split('=', 1)[1])
        elif arg == key:
            if i + 1 >= len(args) or args[i + 1].startswith('--'):
                raise ValueError('missing value for ' + key)
            values.append(args[i + 1])
    if len(values) > 1:
        raise ValueError('repeated flag ' + key)
    return values[0] if values else default


def namespace_for(args):
    job = flag_value(args, 'job_name')
    task = int(flag_value(args, 'task_index', '0'))
    ps = flag_value(args, 'ps_hosts')
    workers = flag_value(args, 'worker_hosts')
    if not ps or len(ps.split(',')) != 1:
        raise ValueError('this launcher requires exactly one explicit --ps_hosts entry')
    if not workers or not all(workers.split(',')):
        raise ValueError('explicit --worker_hosts is required')
    if job == 'ps' and task == 0:
        return 1
    if job == 'worker' and 0 <= task < min(len(workers.split(',')), 32766):
        return task + 2
    raise ValueError('invalid job_name/task_index for this cluster')


def grpc_options(current):
    options = [x for x in current.split(',')
               if x and x.split('=', 1)[0] != 'grpc.enable_retries']
    return ','.join(options + ['grpc.enable_retries=0'])



def asset_directory(environment=None):
    environment = os.environ if environment is None else environment
    configured = environment.get('FFN_IDFIX_ASSET_DIR')
    here = Path(__file__).resolve().parent
    if configured:
        base = Path(configured).resolve()
        candidates = [base, base / 'request_id_audit']
    else:
        candidates = [here / 'request_id_audit', here.parent / 'request_id_audit', here]
    for candidate in candidates:
        if ((candidate / 'libffn_id_audit.so').is_file() and
                (candidate / 'resolve_provider.py').is_file()):
            return candidate
    searched = ', '.join(map(str, candidates))
    raise ValueError('libffn_id_audit.so and resolve_provider.py must be in the '
                     'same directory. Searched: ' + searched)


def main(argv):
    is_child = bool(argv and argv[0] == '--_idfixed_child')
    if is_child:
        argv = argv[1:]
    json_mode = bool(argv and argv[0] == '--_idfixed_json') if is_child else any(
        x == '--config' or x.startswith('--config=') for x in argv)
    if is_child and json_mode:
        argv = argv[1:]
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv else 2
    target = Path(__file__).resolve()
    args = argv
    config_env = {}
    if not is_child:
        args, config_env, dry_run = config_launch(argv)
        if dry_run:
            namespace_for(args)
            print(json.dumps({'arguments': args, 'environment': config_env},
                             indent=2, ensure_ascii=False))
            return 0
    if not target.is_file():
        raise ValueError('FFN training script does not exist: ' + str(target))
    namespace = namespace_for(args)
    root = asset_directory({**os.environ, **config_env})
    library = root / 'libffn_id_audit.so'
    if not library.is_file() or not (root / 'resolve_provider.py').is_file():
        raise ValueError('save this launcher in the existing request_id_audit folder '
                         'used by the successful GPU probe')
    if is_child:
        if os.environ.get('FFN_ID_MODE') != 'counter':
            raise ValueError('child must run in counter mode')
        if os.environ.get('FFN_ID_NAMESPACE') != str(namespace):
            raise ValueError('child namespace mismatch')
        import tensorflow.compat.v1 as tf
        sys.path.insert(0, str(root))
        from resolve_provider import register_provider
        register_provider(tf)
        print('FFN_IDFIX_START', {'target': str(target), 'namespace': namespace,
                                  'tensorflow': tf.__version__,
                                  'audit_dir': os.environ['FFN_ID_AUDIT_DIR']}, flush=True)
        # Keep the original target's imports, flags, checkpoint and training loop.
        sys.path.insert(0, str(target.parent))
        sys.argv = [str(target), *args]
        _run_distributed_training(json_mode=json_mode)
        return 0

    env = os.environ.copy()
    env.update(config_env)
    if 'libffn_id_audit' in env.get('LD_PRELOAD', ''):
        raise ValueError('remove the old audit hook from LD_PRELOAD before launching')
    log_root = Path(env.get('FFN_IDFIX_LOG_ROOT', 'ffn-idfixed-logs')).resolve()
    audit_dir = log_root / ('%d-%d-ns%d' % (time.time_ns(), os.getpid(), namespace))
    audit_dir.mkdir(parents=True, exist_ok=False)
    env.update(FFN_ID_MODE='counter', FFN_ID_NAMESPACE=str(namespace),
               FFN_REQUEST_ID_LIBRARY=str(library), FFN_ID_AUDIT_DIR=str(audit_dir),
               GRPC_MAX_RETRIES='0',
               TF_GRPC_DEFAULT_OPTIONS=grpc_options(env.get('TF_GRPC_DEFAULT_OPTIONS', '')))
    env['LD_PRELOAD'] = str(library) + (':' + env['LD_PRELOAD']
                                        if env.get('LD_PRELOAD') else '')
    print('Launching FFN with ID counter; native audit logs:', audit_dir, flush=True)
    # Fresh process: TF is not imported in this supervisor. No automatic replay
    # of a failed step and no automatic restart/restoration of the chief.
    try:
        return subprocess.call([sys.executable, str(Path(__file__).resolve()),
                                '--_idfixed_child',
                                *(['--_idfixed_json'] if json_mode else []),
                                *args], env=env)
    except KeyboardInterrupt:
        return 130


def _run_distributed_training(json_mode=False):
    import json
    import random
    import time

    import tensorflow.compat.v1 as tf

    # The current FFN training code uses placeholders, sessions, and TF1 queues.
    tf.disable_v2_behavior()

    from absl import app
    from absl import flags

    import train as ffn_train
    from ffn.training.import_util import import_symbol


    FLAGS = flags.FLAGS

    flags.DEFINE_string(
        "job_name",
        None,
        'Distributed job name: "ps" or "worker".',
    )
    flags.DEFINE_integer(
        "task_index",
        0,
        "Task index within the selected job.",
    )
    flags.DEFINE_list(
        "ps_hosts",
        None,
        "Comma-separated parameter-server addresses.",
    )
    flags.DEFINE_list(
        "worker_hosts",
        None,
        "Comma-separated worker addresses.",
    )


    def _validate_worker_flags() -> None:
        required = (
            "train_coords",
            "data_volumes",
            "label_volumes",
            "model_name",
            "model_args",
        )
        missing = [
            name
            for name in required
            if getattr(FLAGS, name, None) in (None, "")
        ]
        if missing:
            raise app.UsageError(
                "Missing worker arguments: " + ", ".join(missing)
            )


    def main(argv: list[str]) -> None:
        del argv

        if FLAGS.job_name not in ("ps", "worker"):
            raise app.UsageError('--job_name must be "ps" or "worker".')

        if not FLAGS.ps_hosts:
            raise app.UsageError("--ps_hosts is required.")

        if not FLAGS.worker_hosts:
            raise app.UsageError("--worker_hosts is required.")

        cluster = tf.train.ClusterSpec(
            {
                "ps": list(FLAGS.ps_hosts),
                "worker": list(FLAGS.worker_hosts),
            }
        )

        server_config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        server_config.gpu_options.allow_growth = True

        server = tf.train.Server(
            cluster,
            job_name=FLAGS.job_name,
            task_index=FLAGS.task_index,
            protocol="grpc",
            config=server_config,
        )

        if FLAGS.job_name == "ps":
            # The parameter server owns the shared variables and remains active.
            server.join()
            return

        _validate_worker_flags()

        # Pass the distributed configuration into the existing FFN train.py.
        FLAGS.master = server.target
        FLAGS.task = FLAGS.task_index
        FLAGS.ps_tasks = len(FLAGS.ps_hosts)

        # Match the task-dependent seeding behavior of the original train.py.
        seed = int(time.time() + FLAGS.task * 24 * 60 * 60)
        random.seed(seed)

        model_class = import_symbol(FLAGS.model_name)

        ffn_train.train_ffn(
            model_class,
            batch_size=FLAGS.batch_size,
            **json.loads(FLAGS.model_args),
        )

    if json_mode:
        # JSON options are serialized as --key=value. Keep only flags accepted
        # by this program; direct CLI invocation retains normal Abseil errors.
        sys.argv = [sys.argv[0], *(arg for arg in sys.argv[1:]
                                  if arg[2:].split('=', 1)[0] in FLAGS)]
    app.run(main)


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except ValueError as e:
        print('FFN launcher error:', e, file=sys.stderr)
        sys.exit(2)
