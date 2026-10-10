//! Abstraction of inline task dispatch, intended to be used in jxl-oxide.
//!
//! [`JxlThreadPool`] is re-exported by `jxl-oxide`.

/// Inline task-dispatch wrapper.
///
/// Every spawn operation runs the given closure in place. This wrapper preserves the dispatch
/// interface used by the decoder for a later task-based implementation.
#[derive(Debug, Clone)]
pub struct JxlThreadPool(JxlThreadPoolImpl);

#[derive(Debug, Clone)]
enum JxlThreadPoolImpl {
    None,
}

/// Inline scope created by the dispatch wrapper.
#[derive(Debug, Copy, Clone)]
pub struct JxlScope<'r, 'scope>(JxlScopeInner<'r, 'scope>);

#[derive(Debug, Copy, Clone)]
enum JxlScopeInner<'r, 'scope> {
    None(std::marker::PhantomData<&'r &'scope ()>),
}

impl JxlThreadPool {
    /// Creates an inline dispatch wrapper without any multithreading capability.
    ///
    /// Every spawn operation runs the closure in the current thread.
    pub const fn none() -> Self {
        Self(JxlThreadPoolImpl::None)
    }

    /// Returns if the thread pool is capable of multithreading.
    pub fn is_multithreaded(&self) -> bool {
        false
    }
}

impl JxlThreadPool {
    /// Runs the given closure inline.
    pub fn spawn(&self, op: impl FnOnce() + Send + 'static) {
        match &self.0 {
            JxlThreadPoolImpl::None => op(),
        }
    }

    /// Creates an inline scope of tasks.
    pub fn scope<'scope, R: Send>(
        &'scope self,
        op: impl for<'r> FnOnce(JxlScope<'r, 'scope>) -> R + Send,
    ) -> R {
        match &self.0 {
            JxlThreadPoolImpl::None => op(JxlScope(JxlScopeInner::None(Default::default()))),
        }
    }

    /// Consumes the `Vec`, and runs a job for each element of the `Vec`.
    pub fn for_each_vec<T: Send>(&self, v: Vec<T>, op: impl Fn(T) + Send + Sync) {
        match &self.0 {
            JxlThreadPoolImpl::None => v.into_iter().for_each(op),
        }
    }

    /// Consumes the `Vec`, and runs a job for each element of the `Vec`.
    pub fn for_each_vec_with<T: Send, U: Send + Clone>(
        &self,
        v: Vec<T>,
        init: U,
        op: impl Fn(&mut U, T) + Send + Sync,
    ) {
        match &self.0 {
            JxlThreadPoolImpl::None => {
                let mut init = init;
                v.into_iter().for_each(|item| op(&mut init, item))
            }
        }
    }

    /// Runs a job for each element of the mutable slice.
    pub fn for_each_mut_slice<'a, T: Send>(
        &self,
        v: &'a mut [T],
        op: impl Fn(&'a mut T) + Send + Sync,
    ) {
        match &self.0 {
            JxlThreadPoolImpl::None => v.iter_mut().for_each(op),
        }
    }

    /// Runs a job for each element of the mutable slice.
    pub fn for_each_mut_slice_with<'a, T: Send, U: Send + Clone>(
        &self,
        v: &'a mut [T],
        init: U,
        op: impl Fn(&mut U, &'a mut T) + Send + Sync,
    ) {
        match &self.0 {
            JxlThreadPoolImpl::None => {
                let mut init = init;
                v.iter_mut().for_each(|item| op(&mut init, item))
            }
        }
    }
}

impl<'scope> JxlScope<'_, 'scope> {
    /// Runs the given closure in the current inline scope.
    pub fn spawn(&self, op: impl for<'r> FnOnce(JxlScope<'r, 'scope>) + Send + 'scope) {
        match self.0 {
            JxlScopeInner::None(_) => op(JxlScope(JxlScopeInner::None(Default::default()))),
        }
    }
}

