mod json;
mod npy;
mod spec;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use std::sync::{Arc, OnceLock};

use jxl_oxide::{FrameBuffer, JxlImage, NullCms, Render, Tasks};

type CheckResult<T> = Result<T, String>;

/// The runtime every decode runs on when --workers is given; inline otherwise.
#[cfg(feature = "nagoya")]
static RUNTIME: OnceLock<nagoya::runtime::Runtime> = OnceLock::new();
static TASKS: OnceLock<Tasks> = OnceLock::new();

fn tasks() -> Tasks {
    TASKS.get().cloned().unwrap_or_else(Tasks::inline)
}

fn render(image: &mut JxlImage, index: usize) -> jxl_oxide::Result<Render> {
    #[cfg(feature = "nagoya")]
    if RUNTIME.get().is_some() {
        return nagoya::block_on(image.render_frame_async(index));
    }
    image.render_frame(index)
}

struct Arguments {
    directory: PathBuf,
    only: Option<String>,
    /// Feed the input through the streaming API in independently allocated chunks of this size.
    chunk: Option<usize>,
    /// Print a hash of every rendered frame instead of comparing against references.
    hash: bool,
    /// Decode on a nagoya runtime with this many workers.
    workers: Option<usize>,
}

#[derive(Default)]
struct Report {
    frames: Option<usize>,
    expected_frames: Option<usize>,
    peak: Vec<f64>,
    rmse: Vec<f64>,
    limits: Vec<spec::Limit>,
    milliseconds: f64,
}

impl Report {
    fn print(&self, name: &str, passed: bool) {
        let actual = optional_count(self.frames);
        let expected = optional_count(self.expected_frames);
        let limits = if self.limits.is_empty() {
            "n/a".to_owned()
        } else {
            format!(
                "[{}]",
                self.limits
                    .iter()
                    .map(|limit| format!("{:.8e}/{:.8e}", limit.peak, limit.rmse))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        println!(
            "{} {} frames={}/{} max_err={} rmse={} limit={} ms={:.3}",
            if passed { "PASS" } else { "FAIL" },
            name,
            actual,
            expected,
            measurements(&self.peak),
            measurements(&self.rmse),
            limits,
            self.milliseconds
        );
    }
}

fn optional_count(value: Option<usize>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |value| value.to_string())
}

fn measurements(values: &[f64]) -> String {
    if values.is_empty() {
        return "n/a".to_owned();
    }
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("{value:.8e}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn arguments() -> CheckResult<Option<Arguments>> {
    let mut args = std::env::args_os().skip(1);
    let mut directory = None;
    let mut only = None;
    let mut chunk = None;
    let mut hash = false;
    let mut workers = None;
    let mut positional = false;
    while let Some(arg) = args.next() {
        if !positional {
            match arg.to_str() {
                Some("--help" | "-h") => return Ok(None),
                Some("--") => {
                    positional = true;
                    continue;
                }
                Some("--only") => {
                    if only.is_some() {
                        return Err("--only was supplied more than once".to_owned());
                    }
                    let name = args
                        .next()
                        .ok_or_else(|| "--only requires a case name".to_owned())?;
                    only = Some(
                        name.into_string()
                            .map_err(|_| "case names must be UTF-8".to_owned())?,
                    );
                    continue;
                }
                Some("--hash") => {
                    hash = true;
                    continue;
                }
                Some("--chunk") => {
                    let size = args
                        .next()
                        .and_then(|value| value.into_string().ok())
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|size| *size > 0)
                        .ok_or_else(|| "--chunk requires a positive byte count".to_owned())?;
                    chunk = Some(size);
                    continue;
                }
                Some("--workers") => {
                    let count = args
                        .next()
                        .and_then(|value| value.into_string().ok())
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|count| *count > 0)
                        .ok_or_else(|| "--workers requires a positive count".to_owned())?;
                    workers = Some(count);
                    continue;
                }
                Some(value) if value.starts_with('-') => {
                    return Err(format!("unknown argument: {value}"));
                }
                _ => {}
            }
        }
        if directory.replace(PathBuf::from(arg)).is_some() {
            return Err("supply exactly one conformance directory".to_owned());
        }
    }
    Ok(Some(Arguments {
        directory: directory
            .ok_or_else(|| "a conformance directory is required".to_owned())?,
        only,
        chunk,
        hash,
        workers,
    }))
}

fn case_directories(args: &Arguments) -> CheckResult<Vec<(String, PathBuf)>> {
    let nested = args.directory.join("testcases");
    let directory = if nested.is_dir() {
        nested
    } else {
        args.directory.clone()
    };
    let entries = std::fs::read_dir(&directory)
        .map_err(|error| format!("cannot list {}: {error}", directory.display()))?;
    let mut cases = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read directory entry: {error}"))?;
        let kind = entry
            .file_type()
            .map_err(|error| format!("cannot inspect {}: {error}", entry.path().display()))?;
        if !kind.is_dir() {
            continue;
        }
        let name: OsString = entry.file_name();
        let name = name
            .into_string()
            .map_err(|_| "case directory names must be UTF-8".to_owned())?;
        if args.only.as_ref().is_none_or(|only| only == &name) {
            cases.push((name, entry.path()));
        }
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    if cases.is_empty() {
        return Err(match &args.only {
            Some(name) => format!("case {name:?} was not found"),
            None => format!("no cases found in {}", directory.display()),
        });
    }
    Ok(cases)
}

fn read(path: &Path) -> CheckResult<Vec<u8>> {
    std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

/// Decodes through the streaming API, each chunk in its own allocation so every boundary is a span boundary.
fn streamed(input: &[u8], size: usize) -> CheckResult<JxlImage> {
    let mut chunks = input.chunks(size).map(Arc::<[u8]>::from);
    let mut uninit = JxlImage::builder().tasks(tasks()).build_uninit();
    let mut image = loop {
        let ended = match chunks.next() {
            Some(chunk) => {
                uninit.feed(chunk).map_err(|error| format!("feed failed: {error}"))?;
                false
            }
            None => {
                uninit.end_of_input();
                true
            }
        };
        match uninit.try_init().map_err(|error| format!("init failed: {error}"))? {
            jxl_oxide::InitializeResult::Initialized(image) => break image,
            jxl_oxide::InitializeResult::NeedMoreData(_) if ended => {
                return Err("input ended before the image header".to_owned());
            }
            jxl_oxide::InitializeResult::NeedMoreData(more) => uninit = more,
        }
    };
    for chunk in chunks {
        image.feed(chunk).map_err(|error| format!("feed failed: {error}"))?;
    }
    image.finalize().map_err(|error| format!("finalize failed: {error}"))?;
    Ok(image)
}

/// Decodes input.jxl and hashes every rendered keyframe (FNV-1a over the f32 bit patterns of all channels).
/// Needs no reference images, so two builds can be compared on every testcase.
fn hash_case(directory: &Path, chunk: Option<usize>) -> CheckResult<String> {
    let input: Arc<[u8]> = Arc::from(read(&directory.join("input.jxl"))?);
    let mut image = match chunk {
        None => JxlImage::builder()
            .tasks(tasks())
            .read_bytes(input)
            .map_err(|error| format!("decode failed: {error}"))?,
        Some(size) => streamed(&input, size)?,
    };
    image.set_cms(NullCms);
    image.set_render_spot_color(false);
    let frames = image.num_loaded_keyframes();
    let mut hashes = Vec::with_capacity(frames);
    for index in 0..frames {
        let render = render(&mut image, index)
            .map_err(|error| format!("frame {index} render failed: {error}"))?;
        let buffer = render.image_all_channels();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for sample in buffer.buf() {
            for byte in sample.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        hashes.push(format!("{}x{}x{}:{hash:016x}", buffer.width(), buffer.height(), buffer.channels()));
    }
    Ok(format!("frames={frames} {}", hashes.join(",")))
}

fn run_case(directory: &Path, chunk: Option<usize>, report: &mut Report) -> CheckResult<()> {
    let metadata_path = directory.join("test.json");
    let metadata = read(&metadata_path)?;
    let metadata = std::str::from_utf8(&metadata)
        .map_err(|error| format!("{} is not UTF-8: {error}", metadata_path.display()))?;
    let metadata = json::parse(metadata)?;
    metadata.object()?;

    let mut references = spec::references(&metadata, directory)?
        .iter()
        .map(|path| npy::Npy::open(path))
        .collect::<CheckResult<Vec<_>>>()?;
    let channels = references[0].channels;
    let mut expected_frames = 0usize;
    for reference in &references {
        if reference.channels != channels {
            return Err("reference files have different channel counts".to_owned());
        }
        expected_frames = expected_frames
            .checked_add(reference.frames)
            .ok_or_else(|| "reference frame count overflows usize".to_owned())?;
    }
    report.expected_frames = Some(expected_frames);
    report.limits = spec::limits(&metadata, channels)?;

    let target_icc = spec::target_icc(&metadata, directory)?;
    let input: Arc<[u8]> = Arc::from(read(&directory.join("input.jxl"))?);

    let start = Instant::now();
    let prepared: CheckResult<JxlImage> = (|| {
        let mut image = match chunk {
            None => JxlImage::builder()
                .tasks(tasks())
                .read_bytes(input)
                .map_err(|error| format!("decode failed: {error}"))?,
            Some(size) => streamed(&input, size)?,
        };
        image.set_cms(NullCms);
        image.set_render_spot_color(false);
        if let Some(profile) = &target_icc {
            image
                .request_icc(profile)
                .map_err(|error| format!("reference ICC request failed: {error}"))?;
        }
        Ok(image)
    })();
    report.milliseconds += start.elapsed().as_secs_f64() * 1000.0;
    let mut image = prepared?;

    let frames = image.num_loaded_keyframes();
    report.frames = Some(frames);
    if !image.is_loading_done() {
        return Err("input did not load completely".to_owned());
    }
    if frames != expected_frames {
        return Err(format!(
            "displayed frame count is {frames}, expected {expected_frames}"
        ));
    }

    let mut index = 0usize;
    let mut failure = None;
    for reference in &mut references {
        for _ in 0..reference.frames {
            let expected = reference.read_frame()?;
            let start = Instant::now();
            let rendered: CheckResult<FrameBuffer> = (|| {
                let render = render(&mut image, index)
                    .map_err(|error| format!("frame {index} render failed: {error}"))?;
                if render.keyframe_index() != index {
                    return Err(format!("frame {index} returned a different keyframe index"));
                }
                Ok(render.image_all_channels())
            })();
            report.milliseconds += start.elapsed().as_secs_f64() * 1000.0;
            let framebuffer = rendered?;

            if framebuffer.width() != reference.width
                || framebuffer.height() != reference.height
                || framebuffer.channels() != channels
            {
                return Err(format!(
                    "frame {index} shape is {}x{}x{}, expected {}x{}x{}",
                    framebuffer.height(),
                    framebuffer.width(),
                    framebuffer.channels(),
                    reference.height,
                    reference.width,
                    channels
                ));
            }
            if framebuffer.buf().len() != expected.len() {
                return Err(format!("frame {index} sample count differs from reference"));
            }

            if report.peak.is_empty() {
                report.peak = vec![0.0; channels];
                report.rmse = vec![0.0; channels];
            }
            let mut peak = vec![0.0f64; channels];
            let mut squared_error = vec![0.0f64; channels];
            for (sample, (&actual, &expected)) in
                framebuffer.buf().iter().zip(&expected).enumerate()
            {
                let channel = sample % channels;
                if !actual.is_finite() || !expected.is_finite() {
                    report.peak[channel] = f64::INFINITY;
                    report.rmse[channel] = f64::INFINITY;
                    return Err(format!(
                        "frame {index}, sample {sample}: nonfinite output or reference"
                    ));
                }
                let error = (f64::from(actual) - f64::from(expected)).abs();
                peak[channel] = peak[channel].max(error);
                squared_error[channel] += error * error;
            }

            let pixels = expected.len() / channels;
            for channel in 0..channels {
                let rmse = (squared_error[channel] / pixels as f64).sqrt();
                report.peak[channel] = report.peak[channel].max(peak[channel]);
                report.rmse[channel] = report.rmse[channel].max(rmse);
                let limit = report.limits[channel];
                if (peak[channel] > limit.peak || rmse > limit.rmse) && failure.is_none() {
                    failure = Some(format!(
                        "frame {index}, channel {channel}: peak={:.8e}, rmse={:.8e}, limits={:.8e}/{:.8e}",
                        peak[channel], rmse, limit.peak, limit.rmse
                    ));
                }
            }
            index += 1;
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn execute(args: &Arguments) -> CheckResult<bool> {
    let cases = case_directories(args)?;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    for (name, directory) in cases {
        if args.hash {
            match hash_case(&directory, args.chunk) {
                Ok(line) => println!("HASH {name} {line}"),
                Err(error) => println!("HASH {name} error={error}"),
            }
            continue;
        }
        // A testcase whose reference objects were not fetched is skipped, not failed.
        if !std::fs::read_dir(&directory)
            .map(|entries| entries.flatten().any(|entry| entry.path().extension().is_some_and(|e| e == "npy")))
            .unwrap_or(false)
        {
            skipped += 1;
            println!("SKIP {name} (no reference_image.npy)");
            continue;
        }
        let mut report = Report::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_case(&directory, args.chunk, &mut report)
        }));
        let result = result.unwrap_or_else(|_| Err("checker or decoder panicked".to_owned()));
        report.print(&name, result.is_ok());
        match result {
            Ok(()) => passed += 1,
            Err(error) => {
                failed += 1;
                eprintln!("FAIL {name}: {error}");
            }
        }
    }
    println!("conformance: {passed} passed, {failed} failed, {skipped} skipped");
    Ok(failed == 0)
}

fn install_executor(workers: Option<usize>) -> CheckResult<()> {
    let Some(count) = workers else {
        return Ok(());
    };
    #[cfg(feature = "nagoya")]
    {
        let runtime = RUNTIME.get_or_init(|| nagoya::runtime::Runtime::new(count));
        let executor = jxl_oxide::NagoyaExecutor(runtime.executor().clone_handle());
        let _ = TASKS.set(Tasks::new(Arc::new(executor)));
        Ok(())
    }
    #[cfg(not(feature = "nagoya"))]
    {
        let _ = count;
        Err("--workers needs the nagoya feature".to_owned())
    }
}

fn main() -> ExitCode {
    let result = match arguments() {
        Ok(None) => {
            println!("Usage: ps-jxl-check <conformance-directory> [--only <name>] [--hash] [--chunk <bytes>] [--workers <count>]");
            return ExitCode::SUCCESS;
        }
        Ok(Some(args)) => install_executor(args.workers).and_then(|()| execute(&args)),
        Err(error) => Err(error),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            Report::default().print("<suite>", false);
            eprintln!("FAIL <suite>: {error}");
            println!("conformance: 0 passed, 1 failed");
            ExitCode::FAILURE
        }
    }
}

