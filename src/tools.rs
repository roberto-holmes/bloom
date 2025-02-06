use anyhow::{Context, Result};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;

/// Helper function to convert [c_char; SIZE] to string
pub fn vk_to_string(raw_string_array: &[c_char]) -> Result<String> {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    Ok(raw_string
        .to_str()
        .context("Failed to convert vulkan raw string.")?
        .to_owned())
}

pub fn read_shader_code(shader_path: &Path) -> Result<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let spv_file = File::open(shader_path)
        .with_context(|| format!("Failed to find spv file at {:?}", shader_path))?;
    let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

    Ok(bytes_code)
}
