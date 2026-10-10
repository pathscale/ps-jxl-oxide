use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use crate::CheckResult;
use crate::json::Value;

#[derive(Clone, Copy)]
pub struct Limit {
    pub peak: f64,
    pub rmse: f64,
}

fn field<'a>(value: &'a Value, names: &[&str]) -> CheckResult<Option<&'a Value>> {
    let mut found = None;
    for name in names {
        if let Some(value) = value.get(name) {
            if found.is_some() {
                return Err(format!("conflicting fields: {}", names.join(", ")));
            }
            found = Some(value);
        }
    }
    Ok(found)
}

fn required<'a>(value: &'a Value, names: &[&str]) -> CheckResult<&'a Value> {
    field(value, names)?.ok_or_else(|| format!("missing {}", names.join(" or ")))
}

fn tolerance(value: &Value) -> CheckResult<f64> {
    let value = value.number()?;
    if !value.is_finite() || value < 0.0 {
        return Err("tolerances must be finite and nonnegative".to_owned());
    }
    Ok(value)
}

fn channel_values(value: &Value, channels: usize) -> CheckResult<Vec<f64>> {
    match value {
        Value::Number(_) => Ok(vec![tolerance(value)?; channels]),
        Value::Array(values) if values.len() == channels => {
            values.iter().map(tolerance).collect()
        }
        Value::Array(values) => Err(format!(
            "tolerance array has {} entries, expected {channels}",
            values.len()
        )),
        _ => Err("expected a scalar or per-channel tolerance array".to_owned()),
    }
}

fn single_limit(value: &Value) -> CheckResult<Limit> {
    match value {
        Value::Object(_) => Ok(Limit {
            peak: tolerance(required(value, &["peak_error", "max_error"])?)?,
            rmse: tolerance(required(value, &["rmse", "max_rmse"])?)?,
        }),
        Value::Array(pair) if pair.len() == 2 => Ok(Limit {
            peak: tolerance(&pair[0])?,
            rmse: tolerance(&pair[1])?,
        }),
        _ => Err("expected a per-channel tolerance object or [peak, rmse]".to_owned()),
    }
}

pub fn limits(metadata: &Value, channels: usize) -> CheckResult<Vec<Limit>> {
    let value = field(metadata, &["thresholds", "tolerances", "error_thresholds"])?
        .unwrap_or(metadata);
    match value {
        Value::Object(_) => {
            let peak = channel_values(
                required(value, &["peak_error", "max_error"])?,
                channels,
            )?;
            let rmse = channel_values(required(value, &["rmse", "max_rmse"])?, channels)?;
            Ok(peak
                .into_iter()
                .zip(rmse)
                .map(|(peak, rmse)| Limit { peak, rmse })
                .collect())
        }
        Value::Array(values) => {
            if values.len() != channels {
                return Err(format!(
                    "channel threshold count is {}, expected {channels}",
                    values.len()
                ));
            }
            let mut limits = Vec::with_capacity(channels);
            for (index, value) in values.iter().enumerate() {
                if let Some(channel) = value.get("channel") {
                    if channel.number()? != index as f64 {
                        return Err("channel thresholds must be in channel order".to_owned());
                    }
                }
                limits.push(single_limit(value)?);
            }
            Ok(limits)
        }
        _ => Err("invalid tolerance container".to_owned()),
    }
}

fn local_path(directory: &Path, name: &str) -> CheckResult<PathBuf> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("expected a relative case filename, got {name:?}"));
    }
    Ok(directory.join(path))
}

fn files(directory: &Path, extension: &str) -> CheckResult<Vec<PathBuf>> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("cannot list {}: {error}", directory.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) == Some(extension)
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort_by(|a, b| {
        natural_order(
            &a.file_name().unwrap_or_default().to_string_lossy(),
            &b.file_name().unwrap_or_default().to_string_lossy(),
        )
    });
    Ok(paths)
}

fn natural_order(a: &str, b: &str) -> Ordering {
    let left = a.as_bytes();
    let right = b.as_bytes();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        if left[i].is_ascii_digit() && right[j].is_ascii_digit() {
            let left_start = i;
            let right_start = j;
            while i < left.len() && left[i].is_ascii_digit() {
                i += 1;
            }
            while j < right.len() && right[j].is_ascii_digit() {
                j += 1;
            }
            let mut left_number = &left[left_start..i];
            let mut right_number = &right[right_start..j];
            while left_number.first() == Some(&b'0') {
                left_number = &left_number[1..];
            }
            while right_number.first() == Some(&b'0') {
                right_number = &right_number[1..];
            }
            let order = left_number
                .len()
                .cmp(&right_number.len())
                .then_with(|| left_number.cmp(right_number));
            if order != Ordering::Equal {
                return order;
            }
        } else {
            let order = left[i].cmp(&right[j]);
            if order != Ordering::Equal {
                return order;
            }
            i += 1;
            j += 1;
        }
    }
    (left.len() - i)
        .cmp(&(right.len() - j))
        .then_with(|| a.cmp(b))
}

fn reference_names(value: &Value) -> CheckResult<Vec<&str>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::String(value) => Ok(vec![value.as_str()]),
        Value::Array(values) => values.iter().map(Value::string).collect(),
        Value::Object(_) => {
            let value = required(value, &["npy", "file", "path"])?;
            reference_names(value)
        }
        _ => Err("invalid reference filename specification".to_owned()),
    }
}

pub fn references(metadata: &Value, directory: &Path) -> CheckResult<Vec<PathBuf>> {
    let specified = field(metadata, &["reference", "references", "reference_image"])?;
    if let Some(value) = specified {
        let names = reference_names(value)?;
        let npy_count = names.iter().filter(|name| name.ends_with(".npy")).count();
        if npy_count != 0 {
            if npy_count != names.len() {
                return Err("reference list mixes NPY filenames and other values".to_owned());
            }
            let paths = names
                .into_iter()
                .map(|name| local_path(directory, name))
                .collect::<CheckResult<Vec<_>>>()?;
            let unique: BTreeSet<_> = paths.iter().collect();
            if unique.len() != paths.len() {
                return Err("duplicate reference filenames".to_owned());
            }
            return Ok(paths);
        }
    }

    let paths = files(directory, "npy")?
        .into_iter()
        .filter(|path| {
            !path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .starts_with("preview")
        })
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Err("no materialized NPY references found".to_owned());
    }
    Ok(paths)
}

fn read_profile(path: &Path) -> CheckResult<Option<Vec<u8>>> {
    let profile = std::fs::read(path)
        .map_err(|error| format!("cannot read target ICC {}: {error}", path.display()))?;
    if profile.is_empty() {
        return Err(format!("target ICC {} is empty", path.display()));
    }
    Ok(Some(profile))
}

pub fn target_icc(metadata: &Value, directory: &Path) -> CheckResult<Option<Vec<u8>>> {
    let mut specified = field(metadata, &["reference_icc", "icc"])?;
    if specified.is_none() {
        if let Some(reference) = metadata.get("reference") {
            specified = reference.get("icc");
        }
    }
    if let Some(value) = specified {
        match value {
            Value::Null => return Ok(None),
            Value::String(name) if name == "skip" => return Ok(None),
            Value::String(name) => {
                let hash = name.len() == 64
                    && name.as_bytes().iter().all(|byte| byte.is_ascii_hexdigit());
                if !hash {
                    return read_profile(&local_path(directory, name)?);
                }
                let materialized = directory.join(format!("{name}.icc"));
                if materialized.is_file() {
                    return read_profile(&materialized);
                }
            }
            _ => return Err("target ICC must be a filename, null, or \"skip\"".to_owned()),
        }
    }

    let conventional = directory.join("reference.icc");
    if conventional.is_file() {
        return read_profile(&conventional);
    }
    let profiles = files(directory, "icc")?;
    match profiles.as_slice() {
        [profile] => read_profile(profile),
        [] => Err("no materialized target ICC found; use explicit null to skip".to_owned()),
        _ => Err("multiple ICC files found; specify reference_icc".to_owned()),
    }
}
