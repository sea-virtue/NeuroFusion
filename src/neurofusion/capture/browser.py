"""Browse recorded trials without opening the sensors or modifying data."""
from bisect import bisect_right
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import font, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .reader import iter_event_batches, rows
from .schema import safe_path

TASK_NAMES = {
    "background": "无接触背景", "static_contact": "静止接触", "slide": "滑动",
    "start_stop": "开始—停止—再开始", "reverse": "反转", "press_release": "按压—保持—释放",
}
DIRECTION_NAMES = {"up": "上", "down": "下", "left": "左", "right": "右", "none": "未标方向"}
SPEED_NAMES = {"slow": "慢", "fast": "快", "unmeasured": "未标速度"}
ALL = "全部任务"
BIN_S = 0.1


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Trial:
    def __init__(self, path, manifest):
        self.path = path
        self.manifest = manifest
        self.errors = []
        self.duration = float(manifest["config"]["seconds"])
        start = manifest.get("host_recording_start_ns")
        if start is None:
            raise ValueError("录制未开始，没有可回放的时间轴")
        self.frames = []
        if manifest["config"].get("gelsight"):
            try:
                self.frames = list(rows(path / "gelsight/frames.jsonl"))
            except (OSError, ValueError) as exc:
                self.errors.append(f"图像索引读取失败：{exc}")
        self.frame_times = [(r["host_received_ns"] - start) / 1e9 for r in self.frames]
        self.events = np.empty((0, 4), dtype=np.int64)
        self.event_times = np.empty(0)
        try:
            batches = [(m, e) for m, e in iter_event_batches(path) if len(e)]
            if batches:
                self.events = np.concatenate([e for _, e in batches])
                self.event_times = np.repeat([(m["host_received_ns"] - start) / 1e9 for m, _ in batches],
                                             [len(e) for _, e in batches])
                if np.any(np.diff(self.event_times) < 0):
                    raise ValueError("事件批次的主机时间倒序")
        except (OSError, ValueError, KeyError) as exc:
            self.errors.append(f"事件读取失败：{exc}")
            self.events = self.events[:0]
            self.event_times = self.event_times[:0]
        self.duration = max(self.duration, max(self.frame_times, default=0),
                            float(self.event_times[-1]) if len(self.event_times) else 0)
        bin_count = max(1, int(np.ceil(self.duration / BIN_S)))
        self.bin_edges = np.arange(bin_count + 1) * BIN_S
        self.counts = np.histogram(self.event_times, bins=self.bin_edges)[0]
        self.prompts = []
        annotation_path = path / "annotations.jsonl"
        if annotation_path.exists():
            try:
                self.prompts = [((r["host_marked_ns"] - start) / 1e9, r["name"])
                                for r in rows(annotation_path)
                                if r["source"] == "timed_instruction_not_measured_motion"]
            except (OSError, ValueError, KeyError) as exc:
                self.errors.append(f"提示记录读取失败：{exc}")
        self.quality = {}
        if (path / "quality.json").exists():
            try:
                self.quality = read_json(path / "quality.json")
            except (OSError, ValueError) as exc:
                self.errors.append(f"质检报告读取失败：{exc}")


class Browser:
    def __init__(self, root, directory):
        self.root = root
        self.trial = None
        self.frame_index = None
        self.playing = False
        self.last_tick = time.monotonic()
        self.records = []
        for path in sorted(Path(directory).expanduser().glob("*/manifest.json")):
            try:
                self.records.append((path.parent, read_json(path), None))
            except (OSError, ValueError) as exc:
                self.records.append((path.parent, {}, str(exc)))
        root.title("NeuroFusion | 双触觉数据质检")
        root.geometry("1160x790")
        root.minsize(980, 690)
        if "Noto Sans CJK SC" in font.families(root):
            font.nametofont("TkDefaultFont").configure(family="Noto Sans CJK SC", size=10)

        body = ttk.Frame(root, padding=10)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        self.filter = tk.StringVar(value=ALL)
        tasks = {m.get("config", {}).get("labels", {}).get("task") or "其他" for _, m, _ in self.records}
        names = {TASK_NAMES.get(task, task) for task in tasks}
        filters = ttk.Combobox(sidebar, textvariable=self.filter, values=[ALL, *sorted(names)],
                               state="readonly", width=22)
        filters.pack(fill="x", pady=(0, 7))
        filters.bind("<<ComboboxSelected>>", self.filter_records)
        scroll = ttk.Scrollbar(sidebar)
        self.listbox = tk.Listbox(sidebar, width=27, exportselection=False, yscrollcommand=scroll.set,
                                  font="TkDefaultFont")
        scroll.configure(command=self.listbox.yview)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.select_trial)

        main = ttk.Frame(body)
        main.pack(side="left", fill="both", expand=True)
        self.title = ttk.Label(main, font=("TkDefaultFont", 13, "bold"))
        self.title.pack(anchor="w")
        self.stats = ttk.Label(main, wraplength=820)
        self.stats.pack(anchor="w", pady=(3, 6))
        panels = ttk.Frame(main)
        panels.pack(fill="both", expand=True)
        panels.rowconfigure(0, weight=1)
        self.pictures = []
        for col, title in enumerate(("GelSight 图像", "GelNeuro 事件 · 红 ON / 蓝 OFF")):
            panels.columnconfigure(col, weight=1, uniform="sensors")
            box = ttk.LabelFrame(panels, text=title, padding=4)
            box.grid(row=0, column=col, sticky="nsew", padx=(0, 5))
            pane = tk.Label(box, bg="#101827", fg="white")
            pane.pack(fill="both", expand=True)
            self.pictures.append(pane)

        self.timeline = tk.Canvas(main, height=109, bg="#172234", highlightthickness=0)
        self.timeline.pack(fill="x", pady=(8, 2))
        self.timeline.bind("<Configure>", lambda _: self.draw_chart())
        self.timeline.bind("<Button-1>", self.seek_chart)
        self.timeline.bind("<B1-Motion>", self.seek_chart)
        self.position = tk.DoubleVar()
        self.slider = ttk.Scale(main, variable=self.position, command=self.seek, from_=0)
        self.slider.pack(fill="x")
        bar = ttk.Frame(main)
        bar.pack(fill="x", pady=5)
        self.play = ttk.Button(bar, text="播放", command=self.toggle)
        self.play.pack(side="left")
        self.speed = tk.StringVar(value="1")
        ttk.Combobox(bar, textvariable=self.speed, values=("0.25", "0.5", "1", "2"),
                     state="readonly", width=5).pack(side="left", padx=5)
        ttk.Label(bar, text="倍速").pack(side="left")
        ttk.Button(bar, text="质检详情", command=self.details).pack(side="right")
        self.position_label = ttk.Label(main, wraplength=820)
        self.position_label.pack(anchor="w")
        ttk.Label(main, text="按主机接收时间回放 · 黄色竖线是操作提示 · 灰色短线是相机帧").pack(anchor="w")
        self.filter_records()
        root.after(33, self.tick)

    def filter_records(self, _=None):
        self.visible = []
        self.listbox.delete(0, "end")
        for number, record in enumerate(self.records, 1):
            path, manifest, error = record
            labels = manifest.get("config", {}).get("labels", {})
            task_code = labels.get("task") or "其他"
            task = TASK_NAMES.get(task_code, task_code)
            if self.filter.get() not in (ALL, task):
                continue
            self.visible.append(record)
            direction = labels.get("direction", "none")
            speed = labels.get("speed_level", "unmeasured")
            condition = " ".join(part for part in (
                DIRECTION_NAMES.get(direction, direction) if direction != "none" else "",
                SPEED_NAMES.get(speed, speed) if speed != "unmeasured" else "",
            ) if part)
            self.listbox.insert("end", f"{number:02d}  {task} {condition} #{labels.get('repetition', '?')}")
        if self.visible:
            self.listbox.selection_set(0)
        self.select_trial()

    def select_trial(self, _=None):
        selection = self.listbox.curselection()
        self.pause()
        self.trial = None
        self.frame_index = None
        self.stats.configure(text="")
        self.position_label.configure(text="")
        self.timeline.delete("all")
        for pane in self.pictures:
            self.show(pane, text="")
        if not selection:
            self.title.configure(text="当前筛选没有采集段")
            return
        path, manifest, error = self.visible[selection[0]]
        self.title.configure(text=path.name)
        try:
            if error:
                raise ValueError(error)
            self.trial = Trial(path, manifest)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.stats.configure(text=f"无法回放：{exc}")
            return
        trial = self.trial
        quality = trial.quality
        verdict = ("通过" if quality.get("passed") else "未通过") if "passed" in quality else "无记录"
        if trial.errors:
            verdict += " · 有读取问题"
        timing = quality.get("frame_timing", {})
        fps = timing.get("observed_hz")
        gap = timing.get("max_gap_ms")
        gap_text = f" · {fps:.1f} 帧/秒 · 最长帧间隔 {gap:.0f} ms" if fps is not None and gap is not None else ""
        self.stats.configure(text=f"文件质检：{verdict}    |    {len(trial.events):,} 个事件 · {len(trial.frames)} 帧{gap_text}")
        self.slider.configure(to=trial.duration)
        self.position.set(0)
        self.draw_chart()
        self.draw_current()

    def details(self):
        if self.trial is None:
            return
        trial = self.trial
        labels = trial.manifest["config"]["labels"]
        issues = trial.errors + trial.quality.get("issues", [])
        text = [f"{trial.path.name}", f"物体：{labels.get('object_id', '未记录')}",
                f"录制状态：{trial.manifest.get('status', '未知')}",
                f"备注：{labels.get('notes') or '无'}",
                "文件质检来自该段保存的 quality.json，可用 inspect 命令重新核查。"]
        text.extend(issues or ["未记录文件错误。"])
        text.extend(trial.quality.get("warnings", []))
        messagebox.showinfo("质检详情", "\n\n".join(text), parent=self.root)

    def show(self, pane, image=None, text=""):
        photo = ImageTk.PhotoImage(image, master=self.root) if image is not None else None
        pane.configure(image=photo or "", text=text)
        pane.photo = photo

    def draw_current(self):
        trial = self.trial
        if trial is None:
            return
        t = self.position.get()
        index = bisect_right(trial.frame_times, t) - 1
        if index != self.frame_index:
            self.frame_index = index
            pane = self.pictures[0]
            if index < 0:
                self.show(pane, text="尚未收到首帧" if trial.frames else "本段没有可读图像")
            else:
                try:
                    with Image.open(safe_path(trial.path, trial.frames[index]["path"])) as source:
                        source.draft("RGB", (390, 300))
                        image = source.convert("RGB")
                        image.thumbnail((390, 300), Image.Resampling.LANCZOS)
                    self.show(pane, image)
                except (OSError, ValueError, KeyError) as exc:
                    self.show(pane, text=f"第 {index + 1} 帧无法读取\n{exc}")
        left, right = np.searchsorted(trial.event_times, (t - BIN_S, t), side="right")
        events = trial.events[left:right]
        raster = np.zeros(128 * 128, dtype=np.uint8)
        valid = events[(events[:, :2] >= 0).all(axis=1) & (events[:, :2] < 128).all(axis=1)
                       & np.isin(events[:, 3], (0, 1))]
        pixels = valid[:, 1] * 128 + valid[:, 0]
        _, last = np.unique(pixels[::-1], return_index=True)
        last = len(pixels) - 1 - last
        raster[pixels[last]] = valid[last, 3] + 1
        palette = np.array([[16, 24, 39], [67, 151, 255], [255, 115, 93]], dtype=np.uint8)
        image = Image.fromarray(palette[raster.reshape(128, 128)]).resize((288, 288), Image.Resampling.NEAREST)
        self.show(self.pictures[1], image)
        prompt_index = bisect_right(trial.prompts, (t, "\U0010ffff")) - 1
        prompt = f"   |   提示 {trial.prompts[prompt_index][1]}" if prompt_index >= 0 else ""
        self.position_label.configure(text=f"{t:.2f} / {trial.duration:.2f} 秒   |   图像 {index + 1}/{len(trial.frames)}"
                                           f"   |   过去 100 ms：{right-left:,} 个事件{prompt}")
        self.timeline.delete("cursor")
        x = self.chart_x(t)
        self.timeline.create_line(x, 16, x, 103, fill="#ff726d", width=2, tags="cursor")

    def chart_x(self, t):
        return 12 + (max(self.timeline.winfo_width(), 25) - 24) * t / self.trial.duration

    def draw_chart(self):
        self.timeline.delete("all")
        trial = self.trial
        if trial is None:
            return
        peak = max(1, int(trial.counts.max()))
        self.timeline.create_text(12, 9, anchor="w", text=f"每 100 ms 的事件数   峰值 {int(trial.counts.max()):,}",
                                  fill="#d7e4fa", font="TkDefaultFont")
        for i, count in enumerate(trial.counts):
            x1, x2 = self.chart_x(trial.bin_edges[i]), self.chart_x(min(trial.duration, trial.bin_edges[i + 1]))
            if count:
                self.timeline.create_rectangle(x1, 83 - max(1, 59 * count / peak), x2, 83,
                                               fill="#59c8e1", outline="")
        for t in trial.frame_times:
            x = self.chart_x(t)
            self.timeline.create_line(x, 88, x, 99, fill="#a3adbc")
        for t, _ in trial.prompts:
            x = self.chart_x(t)
            self.timeline.create_line(x, 20, x, 86, fill="#ffd46b", dash=(3, 3))
        x = self.chart_x(self.position.get())
        self.timeline.create_line(x, 16, x, 103, fill="#ff726d", width=2, tags="cursor")

    def seek_chart(self, event):
        if self.trial:
            fraction = (event.x - 12) / max(self.timeline.winfo_width() - 24, 1)
            self.seek(str(self.trial.duration * min(1, max(0, fraction))))

    def seek(self, value):
        self.pause()
        self.position.set(float(value))
        self.draw_current()

    def pause(self):
        self.playing = False
        self.play.configure(text="播放")

    def toggle(self):
        if self.trial is None:
            return
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self.last_tick = time.monotonic()
            self.play.configure(text="暂停")

    def tick(self):
        now = time.monotonic()
        if self.playing and self.trial is not None:
            elapsed = (now - self.last_tick) * float(self.speed.get())
            self.position.set((self.position.get() + elapsed) % self.trial.duration)
            self.draw_current()
        self.last_tick = now
        self.root.after(33, self.tick)


def run_browser(directory):
    root = tk.Tk()
    Browser(root, directory)
    print("TACTILE_BROWSER_READY", flush=True)
    root.mainloop()
