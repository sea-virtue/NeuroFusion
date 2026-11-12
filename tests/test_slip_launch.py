from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from neurofusion.control.launch import build_commands,main,stop_child
from neurofusion.control.slip import SlipConfig


class LaunchTests(unittest.TestCase):
    def test_commands_set_rates_without_changing_control_configuration(self):
        args=SimpleNamespace(checkpoint=Path('model.pt'),window_ms=1,clock_hz=100.,
                             poll_ms=.25,control_hz=100.,shadow=False)
        config=SlipConfig(grip_closure=.44)
        original=asdict(config)
        source,control=build_commands(args,Path('run'),config)
        self.assertEqual(source[source.index('--window-ms')+1],'1')
        self.assertEqual(control[control.index('--control-hz')+1],'100.0')
        self.assertIn('--execute',control)
        args.shadow=True
        self.assertNotIn('--execute',build_commands(args,Path('run'),config)[1])
        self.assertEqual(asdict(config),original)

    def test_stop_requests_sdk_cleanup_before_any_forceful_termination(self):
        process=Mock()
        process.poll.return_value=None
        self.assertFalse(stop_child(process))
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_bash_entry_works_from_another_directory_without_starting_devices(self):
        script=Path(__file__).resolve().parents[1]/'scripts/slip-catch.sh'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=root/'config.json';config.write_text(json.dumps(asdict(SlipConfig(grip_closure=.44))))
            checkpoint=root/'checkpoint.pt';checkpoint.touch()
            output=root/'not-created'
            result=subprocess.run([str(script),'--dry-run','--config',str(config),
                                   '--checkpoint',str(checkpoint),'--output',str(output)],
                                  cwd=root,text=True,capture_output=True,check=True)
            self.assertIn('100 Hz',result.stdout)
            self.assertIn('--window-ms 1',result.stdout)
            self.assertFalse(output.exists())

    def test_controller_exit_stops_live_publisher_gracefully(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            marker=root/'source-stopped'
            source_code=("import signal,sys,time; from pathlib import Path; "
                         f"signal.signal(signal.SIGINT,lambda *_:(Path({str(marker)!r}).write_text('SIGINT'),sys.exit(0))); "
                         "print('ready',flush=True); time.sleep(30)")
            control_code="import sys; assert sys.stdin.readline().strip()=='ready'"
            config=root/'config.json';config.write_text(json.dumps(asdict(SlipConfig(grip_closure=.44))))
            checkpoint=root/'checkpoint.pt';checkpoint.touch()
            output=root/'run'
            argv=['launch','--config',str(config),'--checkpoint',str(checkpoint),'--output',str(output),'--shadow']
            commands=([sys.executable,'-u','-c',source_code],[sys.executable,'-u','-c',control_code])
            with patch.object(sys,'argv',argv),patch('neurofusion.control.launch.build_commands',return_value=commands),patch('sys.stdout',io.StringIO()):
                self.assertEqual(main(),0)
            report=json.loads((output/'run.json').read_text())
            self.assertEqual(report['controller_exit_code'],0)
            self.assertEqual(report['publisher_exit_code'],0)
            self.assertFalse(report['forced_publisher_stop'])
            self.assertEqual(marker.read_text(),'SIGINT')

    def test_ctrl_c_cleans_up_both_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=root/'config.json';config.write_text(json.dumps(asdict(SlipConfig(grip_closure=.44))))
            checkpoint=root/'checkpoint.pt';checkpoint.touch()
            output=root/'run'
            def child(kind):
                stop=root/(kind+'-stopped');ready=root/(kind+'-ready')
                prefix=("import signal,sys,time; from pathlib import Path; "
                        "signal.signal(signal.SIGINT,lambda *_:sys.exit(0));\n")
                loop=" while True:\n  print('x',flush=True); time.sleep(.01)\n" if kind=='source' else " for line in sys.stdin:\n  time.sleep(.005)\n"
                body=f"try:\n Path({str(ready)!r}).touch()\n"+loop+f"finally:\n Path({str(stop)!r}).write_text('CLEANED')\n"
                return [sys.executable,'-u','-c',prefix+body]
            commands=(child('source'),child('control'))
            driver=root/'driver.py'
            driver.write_text('import neurofusion.control.launch as launch\n'
                              f'launch.build_commands=lambda *args: {commands!r}\n'
                              'raise SystemExit(launch.main())\n')
            environment=os.environ.copy()
            environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
            process=subprocess.Popen([sys.executable,str(driver),'--config',str(config),
                        '--checkpoint',str(checkpoint),'--output',str(output),'--shadow'],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=environment)
            try:
                deadline=time.monotonic()+5
                while not ((root/'source-ready').exists() and (root/'control-ready').exists()):
                    if process.poll() is not None or time.monotonic()>deadline:
                        self.fail('Test children failed to start')
                    time.sleep(.01)
                process.send_signal(signal.SIGINT)
                stdout,stderr=process.communicate(timeout=5)
                self.assertEqual(process.returncode,130,stdout+stderr)
                for kind in ('source','control'):
                    self.assertEqual((root/(kind+'-stopped')).read_text(),'CLEANED')
                report=json.loads((output/'run.json').read_text())
                self.assertFalse(report['forced_controller_stop'])
                self.assertFalse(report['forced_publisher_stop'])
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=5)


if __name__=='__main__':unittest.main()
