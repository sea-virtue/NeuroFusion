"""Apply/inspect/restore volatile Speck2f configurations; no motor interfaces."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


def load_config(path, samna):
    config = samna.speck2f.configuration.SpeckConfiguration()
    config.from_json(Path(path).read_text())
    valid, reason = samna.speck2f.validate_configuration(config)
    if not valid:
        raise ValueError(f'Invalid Speck2f configuration: {reason}')
    return config


def summarize(config):
    return dict(dvs=json.loads(config.dvs_layer.to_json()),
                monitored_cores=[i for i,c in enumerate(config.cnn_layers) if c.monitor_enable],
                destinations=[d.layer for d in config.dvs_layer.destinations if d.enable])


def apply_verified(model, config, samna):
    model.apply_configuration(config)
    actual = model.get_configuration()
    # SDK configuration readback, not an independent electrical RAM read test.
    expected_json, actual_json = json.loads(config.to_json()), json.loads(actual.to_json())
    # Observed Samna 0.48.6 devkit normalization: enabling monitor output sets
    # ioSel bit 5. Accept only this exact 0 -> 32 change, never mask CNN fields.
    requested_io = expected_json.get('value0',{}).get('factoryConfig',{}).get('ioSel')
    actual_io = actual_json.get('value0',{}).get('factoryConfig',{}).get('ioSel')
    if (getattr(samna,'__version__','').startswith('0.48.6')
            and requested_io == 0 and actual_io == 32
            and (config.dvs_layer.raw_monitor_enable or config.dvs_layer.monitor_enable
                 or any(c.monitor_enable for c in config.cnn_layers))):
        expected_json['value0']['factoryConfig']['ioSel'] = 32
    if actual_json != expected_json:
        differences = []
        def compare(a, b, path=''):
            if len(differences) >= 8 or a == b: return
            if isinstance(a,dict) and isinstance(b,dict) and a.keys() == b.keys():
                for k in a: compare(a[k],b[k],path+'/'+k)
            elif isinstance(a,list) and isinstance(b,list) and len(a) == len(b):
                for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
            else:
                differences.append(dict(path=path,expected=repr(a)[:120],actual=repr(b)[:120]))
        compare(expected_json,actual_json)
        raise RuntimeError('SDK configuration readback differs from requested configuration: '+json.dumps(differences))
    valid, reason = samna.speck2f.validate_configuration(actual)
    if not valid:
        raise RuntimeError(f'Readback configuration invalid: {reason}')
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inspect','deploy','restore'))
    p.add_argument('--config', type=Path, help='Exported config JSON; restore accepts before.json')
    p.add_argument('--output', type=Path, required=True, help='New audit directory')
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--self-test', action='store_true', help='Synthetic input to first CNN core; never a tactile accuracy test')
    p.add_argument('--restore-on-exit', action='store_true')
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        p.error('seconds must be in (0,60]')
    if args.action != 'inspect' and args.config is None:
        p.error('--config required for deploy/restore')
    if args.action != 'deploy' and (args.self_test or args.restore_on_exit):
        p.error('self-test and restore-on-exit apply only to deploy')
    import samna
    target = load_config(args.config, samna) if args.config else None
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='starting', action=args.action,
                  utc=datetime.now(timezone.utc).isoformat(), samna=samna.__version__,
                  flash_written=False, motor_commands_sent=0,
                  persistence='Runtime SDK session only; reapply in each new device-owning process',
                  scope='CNN backbone only; host traces and classifier not deployed')
    device = graph = watch = None
    before = None
    changed = success = False
    try:
        devices = [d for d in samna.device.get_unopened_devices()
                   if d.device_type_name == 'Speck2fModuleDevKit']
        if len(devices) != 1:
            raise RuntimeError(f'Expected one unopened Speck2fModuleDevKit, found {len(devices)}')
        report['device'] = str(devices[0])
        device = samna.device.open_device(devices[0])
        model = device.get_model()
        before = model.get_configuration()
        (args.output/'before.json').write_text(before.to_json())
        report['before'] = summarize(before)
        if args.action == 'inspect':
            success = True
        else:
            changed = True  # Also recover from a partially failed apply.
            actual = apply_verified(model, target, samna)
            report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
            report['sdk_readback_verified'] = True
            report['io_sel'] = dict(requested=target.factory_config.io_sel,
                                    actual=actual.factory_config.io_sel,
                                    note='Samna 0.48.6 devkit monitor normalization 0 -> 32 is allowed; all other fields compared exactly')
            report['applied'] = summarize(actual)
            (args.output/'applied.json').write_text(actual.to_json())
            if args.action == 'deploy':
                cores = report['applied']['monitored_cores']
                if len(cores) != 1:
                    raise RuntimeError('Expected exactly one monitored output core')
                graph = samna.graph.EventFilterGraph()
                sink = samna.BasicSinkNode_speck2f_event_output_event()
                graph.sequential([device.get_model_source_node(), sink])
                if args.self_test:
                    source = samna.BasicSourceNode_speck2f_event_input_event()
                    graph.sequential([source, device.get_model_sink_node()])
                graph.start()
                watch = device.get_stop_watch()
                watch.start()
                if args.self_test:
                    # Bypass the sensor/pool for a deterministic CNN-chain smoke test.
                    # The first core expects 2x32x32 pooled events. No timestamp replay
                    # or numerical equivalence/throughput claim is made here.
                    test = load_config(args.config, samna)
                    test.dvs_layer.pass_sensor_events = False
                    test.dvs_layer.raw_monitor_enable = False
                    test.dvs_layer.monitor_enable = False
                    apply_verified(model, test, samna)
                    first_core = report['applied']['destinations'][0]
                    sink.get_events()
                    total = 0
                    for repeat in range(20):
                        batch = []
                        for y in range(32):
                            for x in range(32):
                                event = samna.speck2f.event.Spike()
                                event.layer, event.x, event.y = first_core,x,y
                                event.feature = (x+y+repeat)%2
                                batch.append(event)
                        source.write(batch)
                        time.sleep(.02)
                        total += sum(isinstance(e,samna.speck2f.event.Spike)
                                     and e.layer == cores[0] for e in sink.get_events())
                    time.sleep(.2)
                    total += sum(isinstance(e,samna.speck2f.event.Spike)
                                 and e.layer == cores[0] for e in sink.get_events())
                    report['self_test'] = dict(synthetic=True, input_spikes=20480,
                                              output_spikes=total, passed=total>0,
                                              scope='First CNN core through last CNN core; sensor/pool bypassed')
                    if total == 0:
                        raise RuntimeError('Synthetic CNN-chain smoke test produced no output spikes')
                # Live diagnostic monitors RAW DVS separately from last-core spikes.
                live = load_config(args.config, samna)
                live.dvs_layer.raw_monitor_enable = True
                apply_verified(model, live, samna)
                sink.get_events()
                counts = Counter()
                with (args.output/'live-events.jsonl').open('x') as log:
                    start = time.monotonic()
                    while time.monotonic()-start < args.seconds:
                        for e in sink.get_events():
                            if isinstance(e,samna.speck2f.event.DvsEvent):
                                counts['raw_dvs'] += 1
                                row=dict(type='raw_dvs',x=e.x,y=e.y,p=int(e.p),t=e.timestamp)
                            elif isinstance(e,samna.speck2f.event.Spike):
                                counts[f'core_{e.layer}'] += 1
                                row=dict(type='spike',layer=e.layer,x=e.x,y=e.y,feature=e.feature,t=e.timestamp)
                            else:
                                counts[type(e).__name__] += 1
                                continue
                            log.write(json.dumps(row)+'\n')
                        time.sleep(.002)
                report['live'] = dict(seconds=time.monotonic()-start,counts=dict(counts),
                                      note='Quiet input may yield zero CNN spikes; no accuracy or saturation test')
                # Leave the requested deployment, without temporary RAW monitoring.
                apply_verified(model, target, samna)
            success = True
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        try:
            if changed and (not success or args.restore_on_exit):
                try:
                    apply_verified(model, before, samna)
                    report['restored_before'] = True
                except BaseException as exc:
                    report.update(restore_error=repr(exc),status='failed')
                    raise
            else:
                report['restored_before'] = False
        finally:
            try:
                if watch is not None: watch.stop()
            finally:
                try:
                    if graph is not None: graph.stop()
                finally:
                    if device is not None: samna.device.close_device(device)
                    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
