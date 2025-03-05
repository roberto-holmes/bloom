use std::{ffi::CStr, os::raw::c_void};

use anyhow::{anyhow, Context, Result};
use ash::{ext::debug_utils, vk, Entry};

use crate::{tools, VALIDATION};

pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = String::from_utf8_lossy(CStr::from_ptr((*p_callback_data).p_message).to_bytes());
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::debug!("{} - {}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("{} - {}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("{} - {}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("{} - {}", types, message),

        _ => log::warn!("[Unknown] {} - {:?}", types, message),
    };
    vk::FALSE
}

pub struct ValidationInfo {
    pub is_enable: bool,
    pub required_validation_layers: [&'static str; 2],
}

#[allow(unused)]
pub fn print_available_instance_extensions(entry: &Entry) -> Result<()> {
    log::debug!("Available instance extensions:");
    unsafe {
        let extensions = entry.enumerate_instance_extension_properties(None)?;
        for e in extensions {
            log::debug!(
                "\t{} v{}",
                String::from_utf8_lossy(e.extension_name_as_c_str()?.to_bytes()),
                e.spec_version
            );
        }
    }
    Ok(())
}

pub fn check_validation_layer_support(entry: &ash::Entry) -> Result<()> {
    unsafe {
        let layer_properties = entry
            .enumerate_instance_layer_properties()
            .context("Failed to enumerate Instance Layers Properties")?;

        if layer_properties.len() <= 0 {
            log::error!("No available layers.");
            return Err(anyhow!("No available validation layers"));
        } else {
            // log::debug!("Instance Available Layers: ");
            // for layer in layer_properties.iter() {
            //     let layer_name = tools::vk_to_string(&layer.layer_name)?;
            //     log::debug!("\t{}", layer_name);
            // }
        }

        for required_layer_name in VALIDATION.required_validation_layers.iter() {
            let mut is_layer_found = false;

            for layer_property in layer_properties.iter() {
                let test_layer_name = tools::vk_to_string(&layer_property.layer_name)?;
                if (*required_layer_name) == test_layer_name {
                    is_layer_found = true;
                    break;
                }
            }

            if is_layer_found == false {
                return Err(anyhow!("Required validation layer as not found"));
            }
        }
    }
    Ok(())
}

pub fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING, // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(crate::debug::vulkan_debug_callback))
}

pub struct DebugUtils {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugUtils {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Option<Self>> {
        if VALIDATION.is_enable == false {
            return Ok(None);
        }
        let debug_info = populate_debug_messenger_create_info();
        let loader = debug_utils::Instance::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_info, None)? };

        Ok(Some(Self { loader, messenger }))
    }
}

impl Drop for DebugUtils {
    fn drop(&mut self) {
        log::trace!("Dropping debug utils");
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}
