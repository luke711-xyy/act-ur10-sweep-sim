"""Optional pixel-exact disk storage for large expert observation sequences."""
import mmap
import os
from pathlib import Path
import uuid

import numpy as np
from .dataset import ActDatasetWriter


def discard_image_pages(images):
    seen=set()
    for image in images.values():
        if not isinstance(image,np.memmap) or id(image._mmap) in seen:continue
        seen.add(id(image._mmap));image.flush()
        if hasattr(image._mmap,'madvise'):image._mmap.madvise(mmap.MADV_DONTNEED)
        if hasattr(os,'posix_fadvise'):
            with Path(image.filename).open('rb') as handle:
                os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)


class DiskBackedDatasetWriter(ActDatasetWriter):
    def add_episode(self,episode_id,observations,actions,success,metadata,
                    control_trace=None,causal_trace=None):
        if len(observations)!=len(actions):raise ValueError('one action required per observation')
        values=np.asarray(actions,dtype=np.float32)
        if values.ndim!=2 or values.shape[1]!=4:raise ValueError('actions must have shape (frames,4)')
        def frames():
            for i,(observation,action) in enumerate(zip(observations,values),1):
                yield observation,action
                # The stream writer has finished both image-encoding futures
                # before it requests the next frame from this iterator.
                if i%8==0:discard_image_pages(observation['images'])
            if observations:discard_image_pages(observations[-1]['images'])
        return self.add_episode_stream(episode_id,frames(),success,metadata,
                                       control_trace=control_trace,causal_trace=causal_trace)


class NativeImageSpool:
    def __init__(self,directory,max_frames,flush_every=8):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.max_frames=int(max_frames);self.flush_every=int(flush_every)
        if min(self.max_frames,self.flush_every)<1:raise ValueError('positive frame capacity required')
        self.buffer=None;self.count=0

    def store(self,images):
        views=tuple(images)
        if not views:raise ValueError('image views must not be empty')
        shape=next(iter(images.values())).shape
        if (len(shape)!=3 or shape[2]!=3 or any(
                im.shape!=shape or im.dtype!=np.uint8 for im in images.values())):
            raise ValueError('image shape/dtype must be uniform HWC uint8 RGB')
        if self.buffer is None:
            self.views=views;self.shape=shape
            self.path=self.directory/f'{uuid.uuid4().hex}.rgb'
            self.buffer=np.memmap(self.path,mode='w+',dtype=np.uint8,
                                  shape=(self.max_frames,len(views),*shape))
        if views!=self.views or shape!=self.shape:raise ValueError('image views/shape changed')
        if self.count>=self.max_frames:raise ValueError('image spool frame capacity exceeded')
        result={}
        for i,view in enumerate(views):
            pixels=self.buffer[self.count,i]
            np.copyto(pixels,images[view],casting='no')
            result[view]=pixels
        self.count+=1
        if self.count%self.flush_every==0:
            discard_image_pages(result)
        return result
