//! CUDA graphs of the text layers, one per row bucket: a short pass is about 700 kernel launches whose
//! submission costs more than the GPU work, and a graph replays them in one call.
//!
//! What makes a pass capturable is that nothing in it depends on the request beyond the bucket: the tables
//! (token ids, positions, sequence starts, attention tiles, delta chunks and sequences) live in fixed
//! workspace buffers, the table-driven kernels launch with the bucket's maximum block count and skip padding
//! entries, and the per-token kernels run over the bucket's rows.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::sys as drv;
use cudarc::driver::{CudaStream, DriverError};

use super::CudaError;

pub struct Graphs {
    execs: HashMap<usize, drv::CUgraphExec>,
    stream: Arc<CudaStream>,
}

// The handles are only used on the owning stream, from one thread at a time (the engine is behind a mutex).
unsafe impl Send for Graphs {}

fn check(code: drv::CUresult) -> Result<(), CudaError> {
    if code == drv::cudaError_enum::CUDA_SUCCESS { Ok(()) } else { Err(CudaError::Driver(DriverError(code))) }
}

impl Graphs {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { execs: HashMap::new(), stream }
    }

    pub fn len(&self) -> usize {
        self.execs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.execs.is_empty()
    }


    /// Start recording the stream's work instead of running it.
    pub fn begin(&self) -> Result<(), CudaError> {
        self.stream.context().bind_to_thread()?;
        unsafe { check(drv::cuStreamBeginCapture_v2(self.stream.cu_stream(), drv::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)) }
    }

    /// Stop recording and keep the instantiated graph as the bucket's.
    pub fn end(&mut self, rows: usize) -> Result<(), CudaError> {
        let mut graph: drv::CUgraph = std::ptr::null_mut();
        unsafe {
            check(drv::cuStreamEndCapture(self.stream.cu_stream(), &mut graph))?;
            let mut exec: drv::CUgraphExec = std::ptr::null_mut();
            let instantiated = check(drv::cuGraphInstantiateWithFlags(&mut exec, graph, 0));
            let _ = drv::cuGraphDestroy(graph);
            instantiated?;
            if let Some(old) = self.execs.insert(rows, exec) {
                let _ = drv::cuGraphExecDestroy(old);
            }
        }
        Ok(())
    }

    /// Abandon a recording that failed part way (the stream must leave capture mode).
    pub fn abandon(&self) {
        let mut graph: drv::CUgraph = std::ptr::null_mut();
        unsafe {
            let _ = drv::cuStreamEndCapture(self.stream.cu_stream(), &mut graph);
            if !graph.is_null() {
                let _ = drv::cuGraphDestroy(graph);
            }
        }
    }

    /// Replay the bucket's graph on the stream, if one was captured.
    pub fn launch(&self, rows: usize) -> Result<bool, CudaError> {
        let Some(&exec) = self.execs.get(&rows) else { return Ok(false) };
        self.stream.context().bind_to_thread()?;
        unsafe { check(drv::cuGraphLaunch(exec, self.stream.cu_stream()))? };
        Ok(true)
    }
}

impl Drop for Graphs {
    fn drop(&mut self) {
        for (_, exec) in self.execs.drain() {
            unsafe {
                let _ = drv::cuGraphExecDestroy(exec);
            }
        }
    }
}
