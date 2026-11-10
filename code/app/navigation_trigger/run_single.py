"""Team PT only. File replay or camera input with bounded pending-frame storage."""
import argparse,json,sys,time,threading
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'trigger'))
from single_model import Pipeline

class LatestCamera:
    """Keep one latest frame; never build an old-frame inference backlog."""
    def __init__(self,source):
        self.cap=cv2.VideoCapture(source);self.lock=threading.Lock();self.latest=None
        if not self.cap.isOpened():raise ValueError('Cannot open camera')
        self.stop=False;self.origin=time.monotonic();self.index=0
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        while not self.stop:
            ok,img=self.cap.read()
            if not ok:break
            self.index+=1
            with self.lock:self.latest=(self.index,time.monotonic()-self.origin,img)
        self.stop=True
    def get(self):
        with self.lock:out=self.latest;self.latest=None
        return out
    def close(self):
        self.stop=True;self.cap.release();self.thread.join(timeout=2.)

def main(a):
    if a.device!='cpu':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable. Use --device cpu explicitly for CPU testing.')
    p=Pipeline(a.weights,device=a.device,camera_mode=a.camera_mode,imgsz=a.imgsz,crop=not a.no_crop)
    # Load kernels before opening the camera.
    p.step(np.zeros((480,640,3),np.uint8),1,0.);p.reset()
    cam=LatestCamera(int(a.source)) if a.camera else None
    cap=None if cam else cv2.VideoCapture(a.source)
    if cap is not None and not cap.isOpened():raise ValueError('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS) if cap else None
    stride=max(1,round(fps/a.sample_fps)) if fps else 1
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    fi=0;last=0;dropped=0;latencies=[];origin=time.monotonic();written=0
    try:
        with dest.open('w',encoding='utf-8') as f:
            while True:
                if cam:
                    current=cam.get()
                    if current is None:
                        if cam.stop:break
                        time.sleep(.005);continue
                    fi,t,img=current;dropped+=max(0,fi-last-1);last=fi
                else:
                    if not cap.grab():break
                    fi+=1
                    if (fi-1)%stride:continue
                    t=(fi-1)/fps
                    if a.realtime:
                        elapsed=time.monotonic()-origin
                        if t<elapsed-1/a.sample_fps:dropped+=1;continue
                        if t>elapsed:time.sleep(t-elapsed)
                    ok,img=cap.retrieve()
                    if not ok:raise RuntimeError('Decode failed')
                es,d=p.step(img,fi,t);latencies.append(d['processing_s']);written+=1
                for e in es:
                    e['capture_to_result_ms']=(time.monotonic()-cam.origin-t)*1000 if cam else None
                    f.write(json.dumps(e,ensure_ascii=False)+'\n');f.flush();print(json.dumps(e,ensure_ascii=False),flush=True)
                if written%25==0:print(json.dumps(dict(processed=written,source_time_s=t,dropped=dropped,last_processing_ms=d['processing_s']*1000)),flush=True)
    finally:
        if cam:cam.close()
        if cap:cap.release()
    result=dict(frames=written,dropped=dropped,device=a.device,camera=a.camera,realtime_replay=a.realtime,
        processing_mean_ms=1000*float(np.mean(latencies)) if latencies else None,
        processing_p95_ms=1000*float(np.percentile(latencies,95)) if latencies else None,
        note='Source sampling frequency is not measured inference FPS. Network/VLM/TTS are not included.')
    dest.with_suffix('.run.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--camera',action='store_true');p.add_argument('--camera-mode',choices=['egocentric','fixed'],default='egocentric')
    p.add_argument('--device',default='0');p.add_argument('--imgsz',type=int,choices=[640,960,1280],default=960)
    p.add_argument('--no-crop',action='store_true');p.add_argument('--sample-fps',type=float,default=5.);p.add_argument('--realtime',action='store_true')
    p.add_argument('--weights',default=str(ROOT.parent/'weights/final_10class_best.pt'))
    p.add_argument('--output',default='events.jsonl');main(p.parse_args())
