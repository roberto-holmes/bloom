use crate::structures::{QueueFamilyIndices, SurfaceStuff, SwapChainSupportDetails};
use crate::vulkan::Destructor;
use crate::{
    debug::{check_validation_layer_support, populate_debug_messenger_create_info},
    tools, VALIDATION,
};
use crate::{vulkan, MAX_FRAMES_IN_FLIGHT};
use anyhow::{anyhow, Result};
use ash::vk;
use std::collections::HashSet;
use std::ffi::CString;
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

pub fn create_instance(entry: &ash::Entry, window: &Window) -> Result<vulkan::Instance> {
    match check_validation_layer_support(entry) {
        Err(_) if VALIDATION.is_enable => {
            return Err(anyhow!("Validation layers requested, but not available"));
        }
        _ => {}
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Dev")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"Bloom")
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 3, 0)); // Specify that we want Vulkan version 1.3+

    let mut extension_names =
        ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();

    if VALIDATION.is_enable {
        extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        extension_names.push(ash::ext::layer_settings::NAME.as_ptr());
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }
    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    // Add validation related extensions to the instance (if they are wanted)
    let required_validation_layer_raw_names: Vec<CString> = VALIDATION
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const i8> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

    let validation_feature_enables = [vk::ValidationFeatureEnableEXT::DEBUG_PRINTF];
    let mut validation_features = vk::ValidationFeaturesEXT::default()
        .enabled_validation_features(&validation_feature_enables);

    if VALIDATION.is_enable {
        let mut debug_utils_create_info = populate_debug_messenger_create_info();
        create_info = create_info.push(&mut debug_utils_create_info);
        create_info.pp_enabled_layer_names = enable_layer_names.as_ptr();
        create_info.enabled_layer_count = enable_layer_names.len() as u32;

        create_info = create_info.push(&mut validation_features);
        // We need to return inside this scope so that `debug_utils_create_info` doesn't get dropped before
        return Ok(vulkan::Instance::new(entry, create_info)?);
    }
    Ok(vulkan::Instance::new(entry, create_info)?)
}

fn find_queue_families(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> QueueFamilyIndices {
    let mut queue_family_indices = QueueFamilyIndices::new();
    // Find graphics queue family
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(device) };

    let mut fallback_compute_queue_index = None;
    let mut fallback_transfer_queue_index = None;

    for (family_index, queue_family) in queue_family_properties.iter().enumerate() {
        if queue_family_indices.graphics_family.is_none()
            && queue_family.queue_count > 0
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics_family = Some((family_index as u32, 0));
        }

        // Check if the current queue supports presenting images
        let is_present_supported = unsafe {
            surface_stuff
                .surface_loader
                .get_physical_device_surface_support(
                    device,
                    family_index as u32,
                    surface_stuff.surface,
                )
                .expect("Failed to check device for presentation support")
        };
        if queue_family_indices.present_family.is_none()
            && queue_family.queue_count > 0
            && is_present_supported
        {
            queue_family_indices.present_family = Some((family_index as u32, 0));
        }

        if queue_family_indices.compute_family.is_none()
            && queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && queue_family.queue_count > 0
        {
            for i in 0..queue_family.queue_count {
                if !queue_family_indices.is_index_taken(family_index as u32, i)
                    && queue_family.queue_count >= i
                {
                    queue_family_indices.compute_family = Some((family_index as u32, i));
                    break;
                } else if fallback_compute_queue_index.is_none() {
                    // Queue family supports compute but has been used by another queue
                    //(We don't want to share the queue so we will set this as as fallback value in case we don't find a better alternative)
                    fallback_compute_queue_index = Some((family_index as u32, 0));
                }
            }
        }

        if queue_family_indices.transfer_family.is_none()
            && queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && queue_family.queue_count > 0
        {
            for i in 0..queue_family.queue_count {
                if !queue_family_indices.is_index_taken(family_index as u32, i)
                    && queue_family.queue_count >= i
                {
                    queue_family_indices.transfer_family = Some((family_index as u32, i));
                    break;
                } else if fallback_transfer_queue_index.is_none() {
                    // Queue family supports trasnfer but has been used by another queue
                    //(We don't want to share the queue so we will set this as as fallback value in case we don't find a better alternative)
                    fallback_transfer_queue_index = Some((family_index as u32, 0));
                }
            }
        }

        // We only need to find one queue family that meets our requirements
        if queue_family_indices.is_complete() {
            break;
        }
    }

    if queue_family_indices.compute_family.is_none() {
        queue_family_indices.compute_family = fallback_compute_queue_index;
    }

    if queue_family_indices.transfer_family.is_none() {
        queue_family_indices.transfer_family = fallback_transfer_queue_index;
    }

    queue_family_indices
}

fn is_device_extension_supported(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<bool> {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to get device extension properties.")
    };

    let mut available_extension_names = vec![];

    // log::debug!("Available Device Extensions: ");
    for extension in available_extensions.iter() {
        let extension_name = tools::vk_to_string(&extension.extension_name)?;
        // log::debug!("\t{}, Version: {}", extension_name, extension.spec_version);

        available_extension_names.push(extension_name);
    }

    let device_extensions = [
        ash::khr::swapchain::NAME,
        ash::khr::storage_buffer_storage_class::NAME,
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_tracing_pipeline::NAME,
        ash::khr::buffer_device_address::NAME,
        ash::khr::deferred_host_operations::NAME,
        ash::ext::descriptor_indexing::NAME,
        ash::khr::spirv_1_4::NAME,
        ash::khr::shader_float_controls::NAME,
        ash::khr::shader_non_semantic_info::NAME,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME,
    ];

    // Put the extensions in a format that we can easily remove elements from without needing to check if they are there
    let mut required_extensions = HashSet::new();
    for extension in device_extensions {
        required_extensions.insert(String::from_utf8_lossy(extension.to_bytes()).to_string());
    }

    // Look for the extensions we want in the available ones
    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    if !required_extensions.is_empty() {
        log::error!("Required extensions are not supported. Missing:");
        for e in &required_extensions {
            log::error!("\t{}", e);
        }
    }

    Ok(required_extensions.is_empty())
}

fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> Result<bool> {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let device_type = match device_properties.device_type {
        vk::PhysicalDeviceType::CPU => "CPU",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        vk::PhysicalDeviceType::OTHER => "Unknown",
        _ => return Err(anyhow!("Invalid device type found on the GPU")),
    };

    let device_name = tools::vk_to_string(&device_properties.device_name)?;
    log::debug!(
        "\tDevice Name: {}, id: {}, type: {}",
        device_name,
        device_properties.device_id,
        device_type
    );

    log::debug!(
        "\tAPI Version: {}.{}.{}",
        vk::api_version_major(device_properties.api_version),
        vk::api_version_minor(device_properties.api_version),
        vk::api_version_patch(device_properties.api_version)
    );

    log::debug!("\tSupports {} Queue Families", device_queue_families.len());
    log::debug!("\t\tQueue Count | Graphics, Compute, Transfer, Sparse Binding, Timestamp Bits");
    for queue_family in device_queue_families.iter() {
        let is_graphics_support = if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            "✓"
        } else {
            "x"
        };
        let is_compute_support = if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            "✓"
        } else {
            "x"
        };
        let is_transfer_support = if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
            "✓"
        } else {
            "x"
        };
        let is_sparse_support = if queue_family
            .queue_flags
            .contains(vk::QueueFlags::SPARSE_BINDING)
        {
            "✓"
        } else {
            "x"
        };

        log::debug!(
            "\t\t    {}\t    |     {}   ,    {}   ,    {}    ,       {}       ,       {}",
            queue_family.queue_count,
            is_graphics_support,
            is_compute_support,
            is_transfer_support,
            is_sparse_support,
            queue_family.timestamp_valid_bits
        );
    }

    // Check features from different parts of the Vulkan spec
    let mut features_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut features_rt = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push(&mut features_as)
        .push(&mut features_rt)
        .push(&mut features13)
        .push(&mut features12);
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };

    let indices = find_queue_families(instance, physical_device, surface_stuff);
    let extensions_supported = is_device_extension_supported(instance, physical_device)?;

    let mut is_swap_chain_adequate = false;
    if extensions_supported {
        let swapchain_support_details =
            SwapChainSupportDetails::query(physical_device, surface_stuff);
        is_swap_chain_adequate = !swapchain_support_details.formats.is_empty()
            && !swapchain_support_details.present_modes.is_empty();
    }

    let supported_features = unsafe { instance.get_physical_device_features(physical_device) };

    Ok(indices.is_complete()
        && extensions_supported
        && is_swap_chain_adequate
        && supported_features.sampler_anisotropy == vk::TRUE
        && supported_features.fragment_stores_and_atomics == vk::TRUE
        && device_properties.limits.timestamp_period > 0.0
        && device_properties.limits.timestamp_compute_and_graphics == vk::TRUE // If this is false we could still use it but we need to check the queues we want to use for timestamp_valid_bits
        && features12.timeline_semaphore == vk::TRUE
        && features12.descriptor_indexing == vk::TRUE
        && features12.buffer_device_address == vk::TRUE
        && features13.dynamic_rendering == vk::TRUE
        && features13.synchronization2 == vk::TRUE
        && features_rt.ray_tracing_pipeline == vk::TRUE
        && features_as.acceleration_structure == vk::TRUE)
}

pub fn get_max_image_size(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> u32 {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    device_properties.limits.max_image_dimension2_d
}

pub fn pick_physical_device(
    instance: &ash::Instance,
    surface_stuff: &SurfaceStuff,
) -> Result<vk::PhysicalDevice> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    if physical_devices.len() == 0 {
        return Err(anyhow!("Failed to find GPUs with Vulkan Support"));
    }
    log::debug!("Found {} GPUs with Vulkan support", physical_devices.len());
    for device in physical_devices {
        // TODO: Figure out a better way of selecting GPU (e.g. allowing the user to select or ranking them)
        if is_physical_device_suitable(instance, device, surface_stuff)? {
            return Ok(device);
        }
    }
    Err(anyhow!("Unable to find suitable GPU"))
}

pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> Result<(vulkan::Device, QueueFamilyIndices)> {
    let indices = find_queue_families(instance, physical_device, surface_stuff);

    let mut queue_families = vec![];
    queue_families.push(indices.graphics_family.unwrap());
    queue_families.push(indices.present_family.unwrap());
    queue_families.push(indices.compute_family.unwrap());
    queue_families.push(indices.transfer_family.unwrap());

    let mut queue_priorities = vec![1.0];

    // Create a set to store all the queue families we want in case a single queue has satisfies multiple requirements
    let mut unique_queue_families: HashSet<u32> = HashSet::new();
    for q in &queue_families {
        unique_queue_families.insert(q.0);
        // Set up priorities for as many queues as there are
        queue_priorities.push(0.5);
    }

    let mut queue_create_infos = vec![];
    for queue_family in unique_queue_families {
        let mut queue_count = 0;
        for &q in &queue_families {
            if q.0 == queue_family {
                queue_count += 1;
            }
        }
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities[..queue_count]);
        queue_create_infos.push(queue_create_info);
    }

    let device_features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .shader_int64(true)
        .fragment_stores_and_atomics(true);

    let device_extensions = [
        ash::khr::swapchain::NAME.as_ptr(),
        ash::khr::storage_buffer_storage_class::NAME.as_ptr(),
        ash::khr::acceleration_structure::NAME.as_ptr(),
        ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
        ash::khr::buffer_device_address::NAME.as_ptr(),
        ash::khr::deferred_host_operations::NAME.as_ptr(),
        ash::ext::descriptor_indexing::NAME.as_ptr(),
        ash::khr::spirv_1_4::NAME.as_ptr(),
        ash::khr::shader_float_controls::NAME.as_ptr(),
        ash::khr::shader_non_semantic_info::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME.as_ptr(),
    ];

    let mut as_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut rt_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
        .variable_pointers(true)
        .variable_pointers_storage_buffer(true);
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .descriptor_indexing(true)
        .buffer_device_address(true);
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let mut features = unsafe {
        vk::PhysicalDeviceFeatures2::default()
            .extend(&mut features11)
            .extend(&mut features12)
            .extend(&mut features13)
            .extend(&mut rt_features)
            .extend(&mut as_features)
            .features(device_features)
    };

    let create_info = unsafe {
        vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .extend(&mut features)
    };

    let device = vulkan::Device::new(instance, physical_device, create_info)?;
    Ok((device, indices))
}

pub fn create_shader_module(
    device: &ash::Device,
    code: &Vec<u8>,
) -> Result<Destructor<vk::ShaderModule>> {
    let mut create_info = vk::ShaderModuleCreateInfo::default();
    create_info.code_size = code.len();
    create_info.p_code = code.as_ptr() as *const u32;

    let shader_module = unsafe { device.create_shader_module(&create_info, None) };

    match shader_module {
        Ok(module) => Ok(Destructor::new(
            device,
            module,
            device.fp_v1_0().destroy_shader_module,
        )),
        Err(e) => Err(anyhow!("Failed to create shader module: {}", e)),
    }
}

pub fn copy_buffer(
    device: &ash::Device,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: u64,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    unsafe {
        let copy_region = [vk::BufferCopy::default().size(size)];
        device.cmd_copy_buffer(command_buffer, src, dst, &copy_region);
    }
    end_single_time_command(device, command_pool, queue, command_buffer)?;
    Ok(())
}

pub fn create_uniform_buffer<T>(allocator: &vk_mem::Allocator) -> Result<[vulkan::Buffer; 2]> {
    let size = std::mem::size_of::<T>() as u64;
    Ok([
        vulkan::Buffer::new_mapped(allocator, size, vk::BufferUsageFlags::UNIFORM_BUFFER)?,
        vulkan::Buffer::new_mapped(allocator, size, vk::BufferUsageFlags::UNIFORM_BUFFER)?,
    ])
}

pub fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<Destructor<vk::CommandPool>> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    Ok(Destructor::new(
        device,
        unsafe { device.create_command_pool(&pool_info, None) }?,
        device.fp_v1_0().destroy_command_pool,
    ))
}

pub fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }?;
    Ok(command_buffers)
}

pub fn create_storage_image_pair<'a>(
    device: &ash::Device,
    instance: &ash::Instance,
    allocator: &Arc<vk_mem::Allocator>,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    usage: vk::ImageUsageFlags,
    size: PhysicalSize<u32>,
    destination_stage: vk::PipelineStageFlags,
) -> Result<[vulkan::Image<'a>; 2]> {
    let width = std::cmp::min(size.width, get_max_image_size(instance, physical_device));
    let height = std::cmp::min(size.height, get_max_image_size(instance, physical_device));

    let mut images = Vec::with_capacity(2);

    for i in 0..2 {
        images.push(vulkan::Image::new(
            device,
            allocator,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
            width,
            height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            usage,
            vk::ImageAspectFlags::COLOR,
        )?);

        transition_image_layout(
            device,
            command_pool,
            submit_queue,
            images[i].get(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            destination_stage,
        )?;
    }

    match <Vec<vulkan::Image<'_>> as TryInto<[vulkan::Image<'_>; 2]>>::try_into(images) {
        Ok(v) => Ok(v),
        Err(_) => Err(anyhow!(
            "Failed to convert storage image vector to an array"
        )),
    }
}

pub fn begin_single_time_commands(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)? }[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };
    Ok(command_buffer)
}

pub fn end_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    unsafe { device.end_command_buffer(command_buffer)? };
    let buffers_to_submit = [command_buffer];
    let submit_info = [vk::SubmitInfo::default().command_buffers(&buffers_to_submit)];

    unsafe {
        device.queue_submit(submit_queue, &submit_info, vk::Fence::null())?;
        device.queue_wait_idle(submit_queue)?;
        device.free_command_buffers(command_pool, &buffers_to_submit);
    };
    Ok(())
}

pub fn transition_image_layout(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    source_stage: vk::PipelineStageFlags,
    destination_stage: vk::PipelineStageFlags,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let src_access_mask;
    let dst_access_mask;

    if old_layout == vk::ImageLayout::UNDEFINED
        && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        // source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        // destination_stage = vk::PipelineStageFlags::TRANSFER;
    } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        dst_access_mask = vk::AccessFlags::SHADER_READ;
        // source_stage = vk::PipelineStageFlags::TRANSFER;
        // destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    } else if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::GENERAL {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        // source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        // destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    } else if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        && new_layout == vk::ImageLayout::PRESENT_SRC_KHR
    {
        src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        dst_access_mask = vk::AccessFlags::empty();
        // source_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        // destination_stage = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
    } else if old_layout == vk::ImageLayout::UNDEFINED
        && new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        // source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        // destination_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    } else {
        return Err(anyhow!("Unsupported layout transition!"));
    }

    // TODO: Change to ImageMemoryBarrier2 (VK_KHR_synchronization2)
    let barrier = [vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)];

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            source_stage,
            destination_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    end_single_time_command(device, command_pool, submit_queue, command_buffer)?;
    Ok(())
}

#[allow(unused)]
pub fn create_texture_sampler(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Sampler> {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR) // Linear interpolation if we need to oversample the image (magnifying)
        .min_filter(vk::Filter::LINEAR) // Linear interpolation if we are undersampling (minifying)
        .address_mode_u(vk::SamplerAddressMode::REPEAT) // Repeat the image if we sample past the edge
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(device_properties.limits.max_sampler_anisotropy)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK) // COlour to use if we are clamping to border colour
        .unnormalized_coordinates(false) // false means use [0,1) coordinates instead of [0, width) and [0, height)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);
    Ok(unsafe { device.create_sampler(&create_info, None)? })
}

pub fn prepare_timestamp_queries(
    device: &ash::Device,
) -> Result<(Destructor<vk::QueryPool>, Vec<u64>)> {
    let timestamps = vec![0; 2];
    let query_pool_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(timestamps.len() as u32);

    let timestamps_query_pool = Destructor::new(
        device,
        unsafe { device.create_query_pool(&query_pool_info, None)? },
        device.fp_v1_0().destroy_query_pool,
    );

    Ok((timestamps_query_pool, timestamps))
}

pub fn wait_on_semaphore(
    device: &ash::Device,
    semaphore: vk::Semaphore,
    timestamp_to_wait: u64,
    timeout_ns: u64,
) -> ash::VkResult<()> {
    let semaphores = [semaphore];
    let wait_values = [timestamp_to_wait];
    let wait_info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&wait_values);
    unsafe { device.wait_semaphores(&wait_info, timeout_ns) }
}

pub fn create_queue(device: &ash::Device, queue_index: (u32, u32)) -> vk::Queue {
    unsafe { device.get_device_queue(queue_index.0, queue_index.1) }
}

pub fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);
    Ok(unsafe { device.create_semaphore(&semaphore_info, None) }?)
}

pub fn create_commands_flight_frames(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<(
    Destructor<vk::CommandPool>,
    [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
)> {
    let pool_info = vk::CommandPoolCreateInfo {
        queue_family_index,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    let command_pool = Destructor::new(
        device,
        unsafe { device.create_command_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_command_pool,
    );
    let command_buffers =
        create_command_buffers(device, command_pool.get(), MAX_FRAMES_IN_FLIGHT as u32)?;

    let commands_out = <Vec<vk::CommandBuffer> as TryInto<
        [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    >>::try_into(command_buffers)
    .unwrap();

    Ok((command_pool, commands_out))
}

pub fn create_commands_2_flight_frames(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<(
    Destructor<vk::CommandPool>,
    [vk::CommandBuffer; 2 * MAX_FRAMES_IN_FLIGHT],
)> {
    let pool_info = vk::CommandPoolCreateInfo {
        queue_family_index,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    let command_pool = Destructor::new(
        device,
        unsafe { device.create_command_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_command_pool,
    );
    let command_buffers = create_command_buffers(
        device,
        command_pool.get(),
        (2 * MAX_FRAMES_IN_FLIGHT) as u32,
    )?;

    let commands_out = <Vec<vk::CommandBuffer> as TryInto<
        [vk::CommandBuffer; 2 * MAX_FRAMES_IN_FLIGHT],
    >>::try_into(command_buffers)
    .unwrap();

    Ok((command_pool, commands_out))
}
