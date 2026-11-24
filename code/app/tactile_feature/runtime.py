"""Exclusive tactile mode. Normal navigation is suspended until an explicit exit.

Commands invalidate captured frames and in-flight inference by generation.
A pause, completed arrival or model error stays in tactile mode; no hidden
fallback to obstacle guidance. No image is exported to recording/guardian paths.
"""
from __future__ import annotations
import base64
import io
import re
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .step_guide import StepGuide
from .live_detector import LiveDetector

_MODELS = {}
_MODEL_LOCK = threading.Lock()
MAX_AGE = 3.0


def mask_png(mask):
    if mask is None or not np.any(mask): return None
    img=Image.fromarray(np.uint8(mask)*125)
    img.thumbnail((480,480),Image.Resampling.NEAREST)
    rgba=Image.new('RGBA',img.size,(255,211,56,0));rgba.putalpha(img)
    b=io.BytesIO();rgba.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode('ascii')


def infer(image, device, detector):
    from nav_trigger import _STEP_LOCK
    with _STEP_LOCK, _MODEL_LOCK:
        if device not in _MODELS:
            from .segment_tactile import TactileSegmenter
            weights=Path(__file__).resolve().parents[1]/'weights/yolo11n_tactile.pt'
            if not weights.is_file(): raise FileNotFoundError('Tactile checkpoint missing')
            _MODELS[device]=TactileSegmenter(weights=weights,device=device,threads=2)
        pil=Image.fromarray(np.ascontiguousarray(image[:,:,::-1]))
        return detector.predict(_MODELS[device],pil)


def reset_navigation(pipe):
    """Reset existing temporal state on return; never edit model/rule settings."""
    p=pipe.perception
    if getattr(p,'is_trigger',False) and hasattr(p,'p'):
        from nav_trigger import _STEP_LOCK
        with _STEP_LOCK:
            p.p.reset()
            p._pending=[];p._last_t=None;p._cap_ms=None;p._cap0=None
            p.preview_overlay=None
    pipe.last_track.clear();pipe.last_text.clear()


class TactileMode:
    def __init__(self,device,emit,log=print,predictor=None,interval=.12):
        self.device=str(device);self.emit=emit;self.log=log
        self.predictor=predictor or infer;self.interval=interval
        self.lock=threading.RLock()
        self.selected=False;self.enabled=False;self.session='';self.generation=0
        self.guide=StepGuide();self.detector=LiveDetector()
        self.last_end=-1e9;self.last_result=-1e9;self.last_state='off'
        self.last_received=-1e9

    def snapshot(self):
        with self.lock: return self.generation,self.selected

    def normal_valid(self,generation):
        return not self.selected and generation==self.generation

    def command(self,action,session):
        if action not in {'start','pause','stop','confirm'} or not isinstance(session,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',session): return False
        with self.lock:
            if action=='start':
                if self.selected and self.enabled and session==self.session: return True
                self.selected=True;self.enabled=True;self.session=session
                self.guide=StepGuide();self.detector=LiveDetector();self.last_end=-1e9
                msg=dict(state='searching',text='점자블록 전용 모드예요. 멈춘 상태에서 앞쪽 바닥을 비춰 주세요.')
            elif session!=self.session: return False
            elif action=='confirm':
                if not self.enabled or self.last_state!='near' or time.monotonic()-self.last_received>MAX_AGE: return False
                self.enabled=False
                msg=dict(state='complete',text='직접 확인한 도착을 기록했어요. 기본 안내로 돌아가거나 다시 찾을 수 있어요.')
            elif action=='pause':
                self.enabled=False
                msg=dict(state='paused',text='점자블록 찾기를 멈췄어요. 다시 찾기를 누르면 재개해요.')
            else:
                self.enabled=False;self.selected=False
                msg=dict(state='off',text='기본 안내로 돌아갑니다.')
            self.generation+=1;self.last_state=msg['state']
            self.emit(dict(type='tactile',control=True,session=self.session,**msg))
        return True

    def cancel(self):
        with self.lock:
            self.selected=False;self.enabled=False;self.generation+=1;self.last_state='off'

    def update(self,image,meta=None):
        meta=meta or {}
        with self.lock:
            if not self.selected or not self.enabled or time.monotonic()-self.last_end<self.interval: return None
            generation=self.generation;session=self.session;guide=self.guide;detector=self.detector
            if meta.get('_mode_generation',generation)!=generation:return None
        started=time.monotonic()
        try:
            result=self.predictor(image,self.device,detector)
            mask=np.asarray(result['mask'],dtype=bool)
            if mask.shape!=image.shape[:2]:raise ValueError('Tactile mask shape mismatch')
            with self.lock:
                if not self.enabled or generation!=self.generation:return None
                end=time.monotonic();received=meta.get('_received_at',started)
                if end-received>MAX_AGE:
                    guide.reset()
                    data=dict(state='stale',text='분석이 늦어요. 멈추고 최신 결과를 기다려 주세요.',mask=None,target=None,direction=None)
                else:
                    data=guide.update(mask,end)
                    # Detection display is independent of direction eligibility.
                    # Even a small/unmatched detection remains visible for testing.
                    if mask.any() and data['mask'] is None:
                        data.update(state='detected', text='검출된 영역이 보여요. 방향을 확인할 때까지 잠시 멈춰 주세요.',
                                    target=None, direction=None, action=None)
                data.setdefault('action',None)
                data.pop('mask')
                data['mask_png']=mask_png(None if data['state']=='stale' else mask)
                self.last_state=data['state'];self.last_result=end;self.last_received=received;self.last_end=end
                elapsed=end-started
                self.emit(dict(type='tactile',session=session,W=image.shape[1],H=image.shape[0],capture_ms=meta.get('c'),
                    processing_ms=round(elapsed*1000),confidence=round(result.get('confidence',0),3),
                    verified=bool(result.get('verified',False)),views=result.get('views',1),**data))
                return elapsed
        except Exception as e:
            self.log(f'[tactile] {type(e).__name__}: {e}')
            with self.lock:
                if generation!=self.generation:return None
                self.enabled=False;self.generation+=1;self.last_state='error'
                message='점자블록 분석 오류예요. 서버 창의 [tactile] 오류를 확인해 주세요.'
                if 'out of memory' in str(e).lower():message='GPU 메모리가 부족해 찾기를 멈췄어요. 기본 안내로 돌아갈 수 있어요.'
                self.emit(dict(type='tactile',session=session,control=True,state='error',text=message))
            return None
