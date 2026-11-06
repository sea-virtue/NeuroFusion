"""Small optional WSLg recorder UI; raw acquisition lives in child processes."""
import queue
from io import BytesIO
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .reader import inspect_recording
from .recorder import CaptureSession
from .schema import CaptureConfig, DIRECTIONS, TASKS, TrialLabels


def run_gui(output_root="data/raw/tactile", camera=0, events_only=False):
    root = tk.Tk()
    root.title("NeuroFusion | Raw tactile recorder")
    root.geometry("1120x900")
    body = ttk.Frame(root, padding=10)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="RAW TACTILE RECORDING | no model inference, no robot commands",
              font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
    ttk.Label(body, text="Close other GelNeuro / GelSight viewers first. Labels below describe your intended operation.").pack(anchor="w")
    fields = ttk.Frame(body)
    fields.pack(fill="x", pady=8)
    defaults = TrialLabels(task="background", object_id="object01", direction="none", speed_level="unmeasured", contact_level="none")
    variables = {}
    specs = [("task", TASKS), ("object_id", None), ("direction", DIRECTIONS),
             ("speed_level", ("unmeasured", "slow", "fast")),
             ("contact_level", ("unmeasured", "none", "light", "medium", "deep")),
             ("repetition", None), ("material", None), ("seconds", None)]
    for i, (key, choices) in enumerate(specs):
        row, col = divmod(i, 4)
        value = "8" if key == "seconds" else str(getattr(defaults, key))
        variables[key] = tk.StringVar(value=value)
        box = ttk.Frame(fields)
        box.grid(row=row, column=col, sticky="ew", padx=(0, 14), pady=3)
        ttk.Label(box, text=key).pack(anchor="w")
        widget = ttk.Combobox(box, textvariable=variables[key], values=choices, state="readonly", width=22) if choices else ttk.Entry(box, textvariable=variables[key], width=24)
        widget.pack(fill="x")
    for key in ("direction_convention", "mounting_notes", "notes"):
        row = ttk.Frame(body)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=key + ":", width=23).pack(side="left")
        variables[key] = tk.StringVar(value=getattr(defaults, key))
        ttk.Entry(row, textvariable=variables[key]).pack(side="left", fill="x", expand=True)
    both = tk.BooleanVar(value=not events_only)
    ttk.Checkbutton(body, text="Also record GelSight (recommended); uncheck for events only", variable=both).pack(anchor="w", pady=5)
    state = {"session": None, "checking": False, "closing": False}
    results = queue.Queue()
    status = tk.StringVar(value="Ready. Choose a trial, then Record. Motion is prompted after 2 seconds of stillness.")
    path_text = tk.StringVar(value=f"Dataset root: {output_root}")
    phase = tk.StringVar(value="NOT RECORDING")
    ttk.Label(body, textvariable=phase, font=("TkDefaultFont", 18, "bold")).pack(anchor="w", pady=8)
    ttk.Label(body, textvariable=status, wraplength=1070).pack(anchor="w")
    ttk.Label(body, textvariable=path_text, wraplength=1070).pack(anchor="w", pady=4)
    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=8)

    def verify(path):
        try:
            results.put(inspect_recording(path, save=True))
        except Exception as error:
            results.put({"passed": False, "issues": [str(error)], "events": 0, "frames": 0})

    def record():
        try:
            values = {key: variables[key].get() for key in TrialLabels.__dataclass_fields__}
            values["repetition"] = int(values["repetition"])
            config = CaptureConfig(root=output_root, seconds=float(variables["seconds"].get()),
                                   camera=camera, gelsight=both.get(), labels=TrialLabels(**values))
            config.validate()
            state["session"] = CaptureSession(config).start()
            record_button.configure(state="disabled")
            stop_button.configure(state="normal")
            marker_start.configure(state="normal")
            marker_stop.configure(state="normal")
            phase.set("OPENING SENSORS / WAIT")
            path_text.set(str(state["session"].root))
            status.set("Recording starts when every selected sensor is ready. Wait for HOLD STILL.")
        except Exception as error:
            messagebox.showerror("Cannot start recording", str(error))

    def stop():
        if state["session"]:
            state["session"].stop()

    def mark(name):
        session = state["session"]
        if session and session.started_ns is not None and not session.finished:
            session.mark(name)
            status.set(f"Saved operator marker: {name} (button time, not measured physical onset)")

    record_button = ttk.Button(buttons, text="Record new trial", command=record)
    record_button.pack(side="left", padx=4)
    stop_button = ttk.Button(buttons, text="Stop / keep partial trial", command=stop, state="disabled")
    stop_button.pack(side="left", padx=4)
    marker_start = ttk.Button(buttons, text="Mark: motion starts", command=lambda: mark("motion_start"), state="disabled")
    marker_start.pack(side="left", padx=4)
    marker_stop = ttk.Button(buttons, text="Mark: motion stops", command=lambda: mark("motion_stop"), state="disabled")
    marker_stop.pack(side="left", padx=4)
    previews = ttk.Frame(body)
    previews.pack(fill="both", expand=True)
    images = {}
    for i, sensor in enumerate(("events", "gelsight")):
        box = ttk.LabelFrame(previews, text="GelNeuro events (red=ON, blue=OFF)" if sensor == "events" else "GelSight full frames saved; preview reduced only")
        box.grid(row=0, column=i, sticky="nsew", padx=3)
        images[sensor] = tk.Label(box, bg="#111827", text="Preview available while recording", fg="white")
        images[sensor].pack(fill="both", expand=True)
        previews.columnconfigure(i, weight=1)
    previews.rowconfigure(0, weight=1)

    def close():
        state["closing"] = True
        stop()
        if state["session"] is None and not state["checking"]:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    def update():
        session = state["session"]
        if session is not None:
            for event in session.poll():
                if event["kind"] == "phase":
                    phase.set(event["text"])
                    root.bell()
                elif event["kind"] == "error":
                    status.set(f"ERROR: {event['sensor']}: {event['error']}")
            for sensor, values in session.latest_previews().items():
                if sensor == "events":
                    rgb = np.full((*values.shape, 3), [17, 24, 39], dtype=np.uint8)
                    rgb[values > 0] = [255, 107, 87]
                    rgb[values < 0] = [53, 139, 255]
                    picture = Image.fromarray(rgb).resize((384, 384), Image.Resampling.NEAREST)
                else:
                    picture = Image.open(BytesIO(values.payload))
                    picture.draft("RGB", (530, 420))
                    picture = picture.convert("RGB")
                    picture.thumbnail((530, 420), Image.Resampling.BILINEAR)
                photo = ImageTk.PhotoImage(picture)
                images[sensor].configure(image=photo, text="")
                images[sensor].image = photo
            if session.started_ns is not None and not session.stop_event.is_set():
                elapsed = (time.monotonic_ns() - session.started_ns) / 1e9
                counters = []
                for sensor, info in session.manifest["sensors"].items():
                    stats = info.get("stats", {})
                    counters.append(f"{sensor}: {stats.get('events_written', stats.get('frames_written', 0))} saved")
                status.set(f"{elapsed:.1f} / {session.config.seconds:.1f} s | " + " | ".join(counters))
            if session.finished:
                phase.set(session.phase.upper() + " / CHECKING SAVED FILES")
                state["checking"] = True
                for button in (stop_button, marker_start, marker_stop):
                    button.configure(state="disabled")
                threading.Thread(target=verify, args=(session.root,), daemon=True).start()
                state["session"] = None
        try:
            result = results.get_nowait()
            state["checking"] = False
            good = result["passed"]
            phase.set("SAVED AND VERIFIED" if good else "CHECK FAILED / KEEP FILES FOR DIAGNOSIS")
            status.set(f"Events: {result['events']} | GelSight frames: {result['frames']} | " +
                       ("Choose the next condition and record again." if good else "; ".join(result["issues"])))
            record_button.configure(state="normal")
            if good:
                variables["repetition"].set(str(int(variables["repetition"].get()) + 1))
        except queue.Empty:
            pass
        if state["closing"] and state["session"] is None and not state["checking"]:
            root.destroy()
            return
        root.after(75, update)

    root.after(75, update)
    print("CAPTURE_GUI_READY", flush=True)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
