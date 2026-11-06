"""Raw GelSight viewer for WSLg; close window to release camera."""
import time
import tkinter as tk
from PIL import Image, ImageTk
from neurofusion.adapters.live_devices import GelSightSource


def main():
    root=tk.Tk()
    root.title('GelSight LIVE | press the tactile surface')
    root.geometry('760x650+780+60')
    label=tk.Label(root,bg='#111827')
    label.pack(fill=tk.BOTH,expand=True)
    status=tk.Label(root,text='Opening real GelSight camera...',anchor='w')
    status.pack(fill=tk.X)
    frames=0
    start=time.monotonic()
    closing=False

    def close():
        nonlocal closing
        closing=True
        root.destroy()

    root.protocol('WM_DELETE_WINDOW',close)
    with GelSightSource(0) as camera:
        def update():
            nonlocal frames,closing
            if closing: return
            try:
                _,image=camera.read()
                height,width=image.shape[:2]
                picture=Image.fromarray(image)
                picture.thumbnail((740,590),Image.Resampling.BILINEAR)
                photo=ImageTk.PhotoImage(picture)
                label.configure(image=photo)
                label.image=photo
                frames+=1
                status.configure(text=f'REAL GelSight | {width}x{height} | frames={frames} | {frames/(time.monotonic()-start):.1f} FPS | close to stop')
                if frames==1: print('WINDOW_READY GelSight; first live frame displayed',flush=True)
                root.after(10,update)
            except Exception as error:
                print(repr(error),flush=True)
                status.configure(text=f'Camera error: {error}')
        root.after(0,update)
        root.mainloop()
    print('GelSight window closed; camera released',flush=True)


if __name__=='__main__': main()
