//! Owned task dispatch for jxl-oxide.
//!
//! The caller supplies workers. The inline executor runs the same jobs immediately.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{
    any::Any,
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

/// An owned synchronous codec job.
pub type Job = Box<dyn FnOnce() -> Box<dyn Any + Send> + Send + 'static>;
/// A returning task. Dropping this future should cancel outstanding work.
pub type JobFuture = Pin<Box<dyn Future<Output = Option<Box<dyn Any + Send>>> + Send + 'static>>;

/// A caller-owned executor for returning codec jobs.
pub trait TaskExecutor: Send + Sync + 'static {
    fn spawn(&self, job: Job) -> JobFuture;
}

/// A cheap handle passed through the async decoder.
#[derive(Clone)]
pub struct Tasks(Arc<dyn TaskExecutor>, bool);

impl core::fmt::Debug for Tasks {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tasks").finish_non_exhaustive()
    }
}

/// A task was cancelled or returned an output of the wrong type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl core::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("codec task cancelled or returned an invalid output")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Cancelled {}

struct InlineExecutor;

impl TaskExecutor for InlineExecutor {
    fn spawn(&self, job: Job) -> JobFuture {
        let output = job();
        Box::pin(core::future::ready(Some(output)))
    }
}

impl Tasks {
    /// Runs each job during spawn and returns a ready future.
    pub fn inline() -> Self {
        Self(Arc::new(InlineExecutor), true)
    }

    pub fn new(executor: Arc<dyn TaskExecutor>) -> Self {
        Self(executor, false)
    }

    /// Whether jobs run immediately on the calling thread.
    ///
    /// Inline stages can give one job the final destination allocation.
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.1
    }

    /// Spawns every job before awaiting any, then collects outputs in index order.
    ///
    /// All handles are retained by this future. Cancellation is an error, never a panic.
    pub async fn run_all<T: Send + 'static>(
        &self,
        jobs: Vec<Box<dyn FnOnce() -> T + Send + 'static>>,
    ) -> Result<Vec<T>, Cancelled> {
        let futures: Vec<_> = jobs
            .into_iter()
            .map(|job| {
                self.0
                    .spawn(Box::new(move || Box::new(job()) as Box<dyn Any + Send>))
            })
            .collect();
        let mut outputs = Vec::with_capacity(futures.len());
        let mut cancelled = false;
        for future in futures {
            match future.await.and_then(|value| value.downcast::<T>().ok()) {
                Some(value) => outputs.push(*value),
                None => cancelled = true,
            }
        }
        if cancelled {
            Err(Cancelled)
        } else {
            Ok(outputs)
        }
    }
}

/// A synchronous API attempted to drive a future that suspended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotReady;

impl core::fmt::Display for NotReady {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("render suspended; use the async API with the supplied executor")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NotReady {}

/// Polls an inline render exactly once. It never spins, parks or blocks.
pub fn poll_once<F: Future>(future: F) -> Result<F::Output, NotReady> {
    let mut future = core::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(output) => Ok(output),
        Poll::Pending => Err(NotReady),
    }
}

/// Adapts a host's existing Nagoya executor without creating workers.
#[cfg(feature = "nagoya")]
pub struct NagoyaExecutor(pub nagoya::Executor);

#[cfg(feature = "nagoya")]
impl TaskExecutor for NagoyaExecutor {
    fn spawn(&self, job: Job) -> JobFuture {
        Box::pin(NagoyaJob {
            handle: Some(self.0.spawn(async move { job() })),
        })
    }
}

#[cfg(feature = "nagoya")]
struct NagoyaJob {
    handle: Option<nagoya::JoinHandle<Box<dyn Any + Send>>>,
}

#[cfg(feature = "nagoya")]
impl Future for NagoyaJob {
    type Output = Option<Box<dyn Any + Send>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(handle) = this.handle.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(handle).poll(context) {
            Poll::Ready(output) => {
                this.handle.take();
                Poll::Ready(output)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "nagoya")]
impl Drop for NagoyaJob {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.cancel();
        }
    }
}
