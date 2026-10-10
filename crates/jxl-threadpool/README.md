# jxl-threadpool
This crate provides owned task dispatch for jxl-oxide. Callers supply an executor through `Tasks`,
with an inline executor for synchronous use and an optional adapter for existing Nagoya workers.

