import os
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class PPTAudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPT录音助手")
        self.samplerate = 44100
        self.channels = 1
        self.slide_count = 5
        self.slide_files = {}
        self.processed_files = {}
        self.current_slide_index = None
        self.record_stream = None
        self.record_queue = queue.Queue()
        self.record_frames = []
        self.record_start_ts = None
        self.record_updater_id = None
        self.play_thread = None
        self.play_stop = threading.Event()
        self.play_start_ts = None
        self.play_duration = 0.0
        self.target_total_seconds = 300
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(top, text="页数").pack(side=tk.LEFT)
        self.slide_spin = ttk.Spinbox(top, from_=1, to=200, width=6)
        self.slide_spin.set(self.slide_count)
        self.slide_spin.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="生成", command=self._regen_slides).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="目标总时长(mm:ss)").pack(side=tk.LEFT, padx=12)
        self.total_entry = ttk.Entry(top, width=8)
        self.total_entry.insert(0, self._format_mmss(self.target_total_seconds))
        self.total_entry.pack(side=tk.LEFT)
        self.total_entry.bind("<Return>", lambda e: self._update_summary())
        self.total_entry.bind("<FocusOut>", lambda e: self._update_summary())
        ttk.Button(top, text="当前状态导出", command=self._export_current_state).pack(side=tk.RIGHT)
        ttk.Button(top, text="均匀加速导出", command=self._apply_speedup_and_export).pack(side=tk.RIGHT, padx=6)

        mid = ttk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        cols = ("index", "limit", "orig", "proc")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=12)
        self.tree.heading("index", text="页码")
        self.tree.heading("limit", text="平均上限")
        self.tree.heading("orig", text="原时长")
        self.tree.heading("proc", text="加速时长")
        self.tree.column("index", width=70, anchor=tk.CENTER)
        self.tree.column("limit", width=120, anchor=tk.CENTER)
        self.tree.column("orig", width=120, anchor=tk.CENTER)
        self.tree.column("proc", width=120, anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.tag_configure('over', foreground='red')

        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.record_btn = ttk.Button(right, text="开始录音", command=self._toggle_record)
        self.record_btn.pack(fill=tk.X, pady=4)
        ttk.Separator(right).pack(fill=tk.X, pady=6)
        self.play_btn = ttk.Button(right, text="播放", command=self._toggle_play)
        self.play_btn.pack(fill=tk.X, pady=4)
        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Button(right, text="删除音频", command=self._delete_selected).pack(fill=tk.X, pady=4)
        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="速度(0.50-2.00)").pack(fill=tk.X, pady=2)
        self.speed_spin = ttk.Spinbox(right, from_=0.50, to=2.00, increment=0.05, format="%.2f", width=6)
        self.speed_spin.set("1.00")
        self.speed_spin.pack(fill=tk.X, pady=2)
        ttk.Button(right, text="应用速度", command=self._apply_speed_to_selected).pack(fill=tk.X, pady=4)
        ttk.Button(right, text="恢复原速", command=self._reset_speed_selected).pack(fill=tk.X, pady=4)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=10, pady=8)
        self.summary_label = ttk.Label(bottom, text="总时长 00:00 / 00:00")
        self.summary_label.pack(anchor=tk.W)

        self._regen_slides()
        self._update_summary()

    def _time_stretch(self, y, rate):
        if rate == 1.0:
            return y
        try:
            data = np.asarray(y).reshape(1, -1).astype(np.float32)
            reader = ArrayReader(data)
            writer = ArrayWriter(channels=1)
            tsm = wsola(channels=1, speed=float(rate))
            tsm.run(reader, writer)
            return writer.data.reshape(-1)
        except Exception:
            try:
                return librosa.effects.time_stretch(y, rate=rate)
            except Exception:
                L = len(y)
                L2 = max(1, int(round(L / rate)))
                xp = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float32)
                x2 = np.linspace(0.0, 1.0, L2, endpoint=False, dtype=np.float32)
                y2 = np.interp(x2, xp, y).astype(np.float32)
                return y2

    def _regen_slides(self):
        try:
            self.slide_count = int(self.slide_spin.get())
        except Exception:
            self.slide_count = 1
        for i in self.tree.get_children():
            self.tree.delete(i)
        lim = self._format_mmss(self._per_slide_limit_seconds())
        for i in range(1, self.slide_count + 1):
            self.tree.insert("", tk.END, iid=str(i), values=(i, lim, "未录制", "未录制"))
        self._update_summary()

    def _format_mmss(self, seconds):
        seconds = int(max(0, round(seconds)))
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"

    def _parse_mmss(self, text):
        try:
            parts = text.strip().split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return int(text)
        except Exception:
            return self.target_total_seconds

    def _per_slide_limit_seconds(self):
        try:
            target = self._parse_mmss(self.total_entry.get())
        except Exception:
            target = self.target_total_seconds
        if self.slide_count <= 0:
            return 0
        return target / float(self.slide_count)

    def _get_selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def _start_record(self):
        if self.record_stream is not None:
            return
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showwarning("提示", "请选择一个页码")
            return
        self.current_slide_index = idx
        self.record_queue = queue.Queue()
        self.record_frames = []
        self.record_start_ts = time.time()
        try:
            self.record_stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=self._rec_cb)
            self.record_stream.start()
        except Exception as e:
            self.record_stream = None
            messagebox.showerror("错误", str(e))
            return
        self._schedule_record_update()

    def _rec_cb(self, indata, frames, time_info, status):
        if status:
            pass
        self.record_queue.put(indata.copy())

    def _schedule_record_update(self):
        self._drain_record_queue()
        if self.record_stream is not None:
            elapsed = time.time() - self.record_start_ts
            if self.current_slide_index is not None:
                self.tree.set(str(self.current_slide_index), column="orig", value=self._format_mmss(elapsed))
                if self.current_slide_index not in self.processed_files:
                    self.tree.set(str(self.current_slide_index), column="proc", value=self._format_mmss(elapsed))
            self.record_updater_id = self.root.after(100, self._schedule_record_update)
        else:
            self.record_updater_id = None

    def _drain_record_queue(self):
        while True:
            try:
                block = self.record_queue.get_nowait()
                self.record_frames.append(block)
            except queue.Empty:
                break

    def _stop_record(self):
        if self.record_stream is None:
            return
        try:
            self.record_stream.stop()
            self.record_stream.close()
        finally:
            self.record_stream = None
        self._drain_record_queue()
        if self.record_updater_id is not None:
            self.root.after_cancel(self.record_updater_id)
            self.record_updater_id = None
        if not self.record_frames:
            return
        data = np.concatenate(self.record_frames, axis=0)
        os.makedirs("recordings", exist_ok=True)
        path = os.path.join("recordings", f"slide_{self.current_slide_index}.wav")
        sf.write(path, data, self.samplerate)
        self.slide_files[self.current_slide_index] = path
        dur = len(data) / float(self.samplerate)
        self.tree.set(str(self.current_slide_index), column="orig", value=self._format_mmss(dur))
        p = self.processed_files.pop(self.current_slide_index, None)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
        self.tree.set(str(self.current_slide_index), column="proc", value=self._format_mmss(dur))
        self._update_summary()

    def _play_selected(self):
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showwarning("提示", "请选择一个页码")
            return
        path = self.processed_files.get(idx) or self.slide_files.get(idx)
        if not path or not os.path.exists(path):
            messagebox.showwarning("提示", "该页尚未录音")
            return
        if self.play_thread and self.play_thread.is_alive():
            return
        y, sr = sf.read(path, dtype='float32')
        y = np.asarray(y).squeeze()
        self.play_stop.clear()
        self.play_duration = len(y) / float(sr)
        self.play_start_ts = time.time()
        self.play_thread = threading.Thread(target=self._play_worker, args=(y, sr), daemon=True)
        self.play_thread.start()

    def _play_worker(self, y, sr):
        try:
            try:
                sd.play(y, sr, blocking=False)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"播放失败: {e}"))
                return
            start = time.time()
            while not self.play_stop.is_set():
                if time.time() - start >= self.play_duration:
                    break
                time.sleep(0.05)
        finally:
            sd.stop()
            self.root.after(0, self._on_play_finished)

    

    def _stop_play(self):
        self.play_stop.set()
        sd.stop()
        if hasattr(self, 'play_btn'):
            self.play_btn.configure(text="播放")

    def _toggle_record(self):
        if self.record_stream is None:
            self._start_record()
            if self.record_stream is not None and hasattr(self, 'record_btn'):
                self.record_btn.configure(text="停止录音")
        else:
            self._stop_record()
            if hasattr(self, 'record_btn'):
                self.record_btn.configure(text="开始录音")

    def _toggle_play(self):
        if self.play_thread and self.play_thread.is_alive():
            self._stop_play()
            if hasattr(self, 'play_btn'):
                self.play_btn.configure(text="播放")
        else:
            self._play_selected()
            if self.play_thread and self.play_thread.is_alive() and hasattr(self, 'play_btn'):
                self.play_btn.configure(text="停止播放")

    def _on_play_finished(self):
        if hasattr(self, 'play_btn'):
            self.play_btn.configure(text="播放")

    def _delete_selected(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        p1 = self.slide_files.pop(idx, None)
        p2 = self.processed_files.pop(idx, None)
        if p1 and os.path.exists(p1):
            try:
                os.remove(p1)
            except Exception:
                pass
        if p2 and os.path.exists(p2):
            try:
                os.remove(p2)
            except Exception:
                pass
        self.tree.set(str(idx), column="orig", value="未录制")
        self.tree.set(str(idx), column="proc", value="未录制")
        self._update_summary()

    def _apply_speed_to_selected(self):
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showwarning("提示", "请选择一个页码")
            return
        src = self.slide_files.get(idx)
        if not src or not os.path.exists(src):
            messagebox.showwarning("提示", "该页尚未录音")
            return
        try:
            rate = float(self.speed_spin.get())
        except Exception:
            rate = 1.0
        rate = max(0.5, min(2.0, rate))
        y, sr = sf.read(src, dtype='float32')
        y = np.asarray(y).squeeze()
        y2 = self._time_stretch(y, rate=rate)
        os.makedirs("recordings", exist_ok=True)
        dst = os.path.join("recordings", f"slide_{idx}_proc.wav")
        sf.write(dst, y2, sr)
        self.processed_files[idx] = dst
        dur = len(y2) / float(sr)
        self.tree.set(str(idx), column="proc", value=self._format_mmss(dur))
        self._update_summary()

    def _reset_speed_selected(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        p = self.processed_files.pop(idx, None)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
        orig = self.slide_files.get(idx)
        if orig and os.path.exists(orig):
            with sf.SoundFile(orig) as f:
                dur = len(f) / float(f.samplerate)
            self.tree.set(str(idx), column="proc", value=self._format_mmss(dur))
        else:
            self.tree.set(str(idx), column="orig", value="未录制")
            self.tree.set(str(idx), column="proc", value="未录制")
        self._update_summary()

    def _collect_durations(self, mode="effective"):
        durs = []
        for i in range(1, self.slide_count + 1):
            if mode == "orig":
                path = self.slide_files.get(i)
            elif mode == "proc":
                path = self.processed_files.get(i) or self.slide_files.get(i)
            else:  # effective
                path = self.processed_files.get(i) or self.slide_files.get(i)
            if path and os.path.exists(path):
                with sf.SoundFile(path) as f:
                    dur = len(f) / float(f.samplerate)
                durs.append(dur)
            else:
                durs.append(0.0)
        return durs

    def _update_summary(self):
        orig_total = sum(self._collect_durations("orig"))
        proc_total = sum(self._collect_durations("proc"))
        target = self._parse_mmss(self.total_entry.get())
        self.summary_label.configure(text=f"原总时长 {self._format_mmss(orig_total)} | 加速总时长 {self._format_mmss(proc_total)} / 目标 {self._format_mmss(target)}")
        limit_sec = self._per_slide_limit_seconds()
        eff = self._collect_durations("effective")
        lim_str = self._format_mmss(limit_sec)
        for i in range(1, self.slide_count + 1):
            self.tree.set(str(i), column="limit", value=lim_str)
            d = eff[i - 1] if i - 1 < len(eff) else 0.0
            if d > limit_sec and d > 0:
                self.tree.item(str(i), tags=('over',))
            else:
                self.tree.item(str(i), tags=())

    def _speed_factors(self, durations, target_total):
        total = sum(durations)
        n = len(durations)
        if total <= target_total or total == 0 or n == 0:
            return [1.0 for _ in durations]
        avg_target = target_total / float(n)
        excess = [max(0.0, d - avg_target) for d in durations]
        sum_excess = sum(excess)
        if sum_excess == 0:
            scale = target_total / total
            return [1.0 / scale for _ in durations]
        over = total - target_total
        shrink = [over * (e / sum_excess) for e in excess]
        target = [max(0.001, d - s) for d, s in zip(durations, shrink)]
        factors = []
        for d, t in zip(durations, target):
            if d <= avg_target:
                factors.append(1.0 if d > 0 else 1.0)
            else:
                f = d / t if t > 0 else 1.0
                factors.append(min(2.0, max(1.0, f)))
        return factors

    def _apply_speedup_and_export(self):
        try:
            self.target_total_seconds = self._parse_mmss(self.total_entry.get())
        except Exception:
            pass
        durs = self._collect_durations("orig")
        total = sum(durs)
        if total == 0:
            messagebox.showwarning("提示", "尚未有录音")
            return
        if total > self.target_total_seconds:
            uni = total / float(self.target_total_seconds)
        else:
            uni = 1.0
        outdir = filedialog.askdirectory(title="选择导出目录")
        if not outdir:
            return
        os.makedirs(outdir, exist_ok=True)
        combined = []
        for i in range(1, self.slide_count + 1):
            src = self.slide_files.get(i)
            if not src or not os.path.exists(src):
                continue
            y, sr = sf.read(src, dtype='float32')
            y = np.asarray(y).squeeze()
            y2 = self._time_stretch(y, rate=uni) if uni != 1.0 else y
            dst = os.path.join(outdir, f"slide_{i}.wav")
            sf.write(dst, y2, sr)
            self.processed_files[i] = dst
            dur2 = len(y2) / float(sr)
            self.tree.set(str(i), column="proc", value=self._format_mmss(dur2))
            combined.append(y2)
        if combined:
            merged = np.concatenate(combined)
            merged_path = os.path.join(outdir, "combined.wav")
            sf.write(merged_path, merged, self.samplerate)
            messagebox.showinfo("完成", f"已导出到: {outdir}")
        else:
            messagebox.showwarning("提示", "没有可导出的音频")
        self._update_summary()

    def _export_current_state(self):
        outdir = filedialog.askdirectory(title="选择导出目录")
        if not outdir:
            return
        os.makedirs(outdir, exist_ok=True)
        combined = []
        for i in range(1, self.slide_count + 1):
            src = self.processed_files.get(i) or self.slide_files.get(i)
            if not src or not os.path.exists(src):
                continue
            y, sr = sf.read(src, dtype='float32')
            y = np.asarray(y).squeeze()
            dst = os.path.join(outdir, f"slide_{i}.wav")
            sf.write(dst, y, sr)
            combined.append(y)
        if combined:
            merged = np.concatenate(combined)
            merged_path = os.path.join(outdir, "combined.wav")
            sf.write(merged_path, merged, self.samplerate)
            messagebox.showinfo("完成", f"已导出到: {outdir}")
        else:
            messagebox.showwarning("提示", "没有可导出的音频")

def main():
    root = tk.Tk()
    app = PPTAudioRecorderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
