use crate::camera::{Camera, CameraUniforms};
use crate::structures::{
    QueueFamilyIndices, SurfaceStuff, SwapChainStuff, SwapChainSupportDetails,
};
use crate::tools::read_shader_code;
use crate::{
    bvh, material, primitives, IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT, IDEAL_RADIANCE_IMAGE_SIZE_WIDTH,
    MAX_FRAMES_IN_FLIGHT, MAX_MATERIAL_COUNT, MAX_OBJECT_COUNT, MAX_QUAD_COUNT, MAX_SPHERE_COUNT,
    MAX_TRIANGLE_COUNT,
};
use crate::{
    debug::{check_validation_layer_support, populate_debug_messenger_create_info},
    tools, VALIDATION,
};
use anyhow::{anyhow, Context, Result};
use ash::{ext::debug_utils, vk};
use bytemuck::{Pod, Zeroable};
use image;
use memoffset::offset_of;
use std::collections::HashSet;
use std::ffi::CString;
use std::path::Path;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
}

// Vertices required to cover the viewport
const VERTICES: [Vertex; 4] = [
    Vertex {
        pos: [-1.0, -1.0, 0.0],
    },
    Vertex {
        pos: [1.0, -1.0, 0.0],
    },
    Vertex {
        pos: [1.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-1.0, 1.0, 0.0],
    },
];

pub const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct UniformBufferObject {
    pub camera: CameraUniforms,
    pub frame_num: u32,
    pub width: u32,
    pub height: u32,
}

impl UniformBufferObject {
    pub fn new() -> Self {
        Self {
            camera: CameraUniforms::zeroed(),
            frame_num: 0,
            width: 0,
            height: 0,
        }
    }
    pub fn update(&mut self, extent: vk::Extent2D) -> Self {
        self.width = extent.width;
        self.height = extent.height;
        *self
    }
    pub fn tick(&mut self) {
        self.frame_num += 1;
    }
    pub fn reset_samples(&mut self) {
        self.frame_num = 0;
    }
    pub fn update_camera(&mut self, camera: &Camera) {
        self.camera = *camera.uniforms();
    }
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Self, pos) as u32)]
    }
}

pub fn create_instance(entry: &ash::Entry, window: &Window) -> Result<ash::Instance> {
    match check_validation_layer_support(entry) {
        Err(_) if VALIDATION.is_enable => {
            return Err(anyhow!("Validation layers requested, but not available"));
        }
        _ => {}
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Bloom")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No engine")
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 3, 0)); // Specify that we want Vulkan version 1.3+

    let mut extension_names =
        ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();
    extension_names.push(debug_utils::NAME.as_ptr());

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

    let mut debug_utils_create_info = populate_debug_messenger_create_info();
    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

    if VALIDATION.is_enable {
        create_info = create_info.push(&mut debug_utils_create_info);
        create_info.pp_enabled_layer_names = enable_layer_names.as_ptr();
        create_info.enabled_layer_count = enable_layer_names.len() as u32;
    }
    Ok(unsafe { entry.create_instance(&create_info, None) }?)
}

fn find_queue_families(
    instance: &ash::Instance,
    device: &vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> QueueFamilyIndices {
    let mut queue_family_indices = QueueFamilyIndices::new();
    // Find graphics queue family
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(*device) };

    let mut family_index = 0;
    for queue_family in queue_family_properties {
        if queue_family.queue_count > 0
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics_family = Some(family_index);
        }

        // Check if the current queue supports presenting images
        let is_present_supported = unsafe {
            surface_stuff
                .surface_loader
                .get_physical_device_surface_support(
                    *device,
                    family_index as u32,
                    surface_stuff.surface,
                )
                .expect("Failed to check device for presentation support")
        };
        if queue_family.queue_count > 0 && is_present_supported {
            queue_family_indices.present_family = Some(family_index);
        }

        // We only need to find one queue family that meets our requirements
        if queue_family_indices.is_complete() {
            break;
        }

        family_index += 1;
    }

    queue_family_indices
}

fn is_device_extension_supported(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> Result<bool> {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(*physical_device)
            .expect("Failed to get device extension properties.")
    };

    let mut available_extension_names = vec![];

    log::debug!("Available Device Extensions: ");
    for extension in available_extensions.iter() {
        let extension_name = tools::vk_to_string(&extension.extension_name)?;
        log::debug!("\t{}, Version: {}", extension_name, extension.spec_version);

        available_extension_names.push(extension_name);
    }

    let device_extensions = [
        ash::khr::swapchain::NAME,
        ash::khr::storage_buffer_storage_class::NAME,
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
    physical_device: &vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> Result<bool> {
    let device_properties = unsafe { instance.get_physical_device_properties(*physical_device) };
    let device_queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

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

    // Check 1.3 features
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push(&mut features13);
    unsafe { instance.get_physical_device_features2(*physical_device, &mut features) };

    let indices = find_queue_families(instance, physical_device, surface_stuff);
    let extensions_supported = is_device_extension_supported(instance, physical_device)?;

    let mut is_swap_chain_adequate = false;
    if extensions_supported {
        let swapchain_support_details =
            SwapChainSupportDetails::query(physical_device, surface_stuff);
        is_swap_chain_adequate = !swapchain_support_details.formats.is_empty()
            && !swapchain_support_details.present_modes.is_empty();
    }

    let supported_features = unsafe { instance.get_physical_device_features(*physical_device) };

    Ok(indices.is_complete()
        && extensions_supported
        && is_swap_chain_adequate
        && supported_features.sampler_anisotropy == vk::TRUE
        && supported_features.fragment_stores_and_atomics == vk::TRUE
        && device_properties.limits.timestamp_period > 0.0
        && device_properties.limits.timestamp_compute_and_graphics == vk::TRUE // If this is false we could still use it but we need to check the queues we want to use for timestamp_valid_bits
        && features13.dynamic_rendering == vk::TRUE
        && features13.synchronization2 == vk::TRUE)
}

fn get_max_image_size(instance: &ash::Instance, physical_device: &vk::PhysicalDevice) -> u32 {
    let device_properties = unsafe { instance.get_physical_device_properties(*physical_device) };
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
        if is_physical_device_suitable(instance, &device, surface_stuff)? {
            return Ok(device);
        }
    }
    Err(anyhow!("Unable to find suitable GPU"))
}

pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> Result<(ash::Device, QueueFamilyIndices)> {
    let indices = find_queue_families(instance, &physical_device, surface_stuff);

    // Create a set to store all the queue families we want in case a single queue has satisfies multiple requirements
    let mut unique_queue_families = HashSet::new();
    unique_queue_families.insert(indices.graphics_family.unwrap());
    unique_queue_families.insert(indices.present_family.unwrap());

    let queue_priorities = [1.0_f32];
    let mut queue_create_infos = vec![];
    for queue_family in unique_queue_families {
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        queue_create_infos.push(queue_create_info);
    }

    let device_features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .fragment_stores_and_atomics(true);

    let device_extensions = [
        ash::khr::swapchain::NAME.as_ptr(),
        ash::khr::storage_buffer_storage_class::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME.as_ptr(),
    ];

    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let mut features = unsafe {
        vk::PhysicalDeviceFeatures2::default()
            .extend(&mut features13)
            .features(device_features)
    };

    let create_info = unsafe {
        vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .extend(&mut features)
    };

    let device = unsafe { instance.create_device(*physical_device, &create_info, None)? };
    Ok((device, indices))
}

fn create_image_view(
    device: &ash::Device,
    image: &vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let create_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .image(*image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect_flags,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    Ok(unsafe { device.create_image_view(&create_info, None)? })
}

pub fn create_image_views(
    device: &ash::Device,
    surface_format: vk::Format,
    swapchain_images: &Vec<vk::Image>,
) -> Result<Vec<vk::ImageView>> {
    let mut image_views = vec![];
    for image in swapchain_images {
        image_views.push(create_image_view(
            device,
            image,
            surface_format,
            vk::ImageAspectFlags::COLOR,
        )?);
    }
    Ok(image_views)
}

fn create_shader_module(device: &ash::Device, code: &Vec<u8>) -> Result<vk::ShaderModule> {
    let mut create_info = vk::ShaderModuleCreateInfo::default();
    create_info.code_size = code.len();
    create_info.p_code = code.as_ptr() as *const u32;

    let shader_module = unsafe { device.create_shader_module(&create_info, None) };

    match shader_module {
        Ok(module) => Ok(module),
        Err(e) => Err(anyhow!("Failed to create shader module: {}", e)),
    }
}

pub fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let image_layout_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let storage_image_layout_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let ubo_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let material_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bvh_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(4)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let sphere_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(5)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let quad_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(6)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let triangle_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(7)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = [
        image_layout_binding,
        storage_image_layout_binding,
        ubo_layout_bindings,
        material_layout_bindings,
        bvh_layout_bindings,
        sphere_layout_bindings,
        quad_layout_bindings,
        triangle_layout_bindings,
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    Ok(unsafe { device.create_descriptor_set_layout(&layout_info, None) }?)
}

pub fn create_descriptor_sets(
    device: &ash::Device,
    pool: &vk::DescriptorPool,
    set_layout: &vk::DescriptorSetLayout,
    uniforms_buffers: &Vec<vk::Buffer>,
    material_buffers: &Vec<vk::Buffer>,
    bvh_buffer: vk::Buffer,
    spheres_buffer: vk::Buffer,
    quads_buffer: vk::Buffer,
    triangles_buffer: vk::Buffer,
    radiance_image_views: &[vk::ImageView; 2],
) -> Result<Vec<vk::DescriptorSet>> {
    let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        layouts.push(*set_layout);
    }

    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(*pool)
        .set_layouts(layouts.as_slice());

    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info)? };

    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniforms_buffers[i])
            .offset(0)
            .range(std::mem::size_of::<UniformBufferObject>() as u64)];

        let material_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(material_buffers[i])
            .offset(0)
            .range((std::mem::size_of::<material::Material>() * MAX_MATERIAL_COUNT) as u64)];

        let bvh_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(bvh_buffer)
            .offset(0)
            .range((std::mem::size_of::<bvh::AABB>() * MAX_OBJECT_COUNT) as u64)];

        let sphere_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(spheres_buffer)
            .offset(0)
            .range((std::mem::size_of::<primitives::Sphere>() * MAX_SPHERE_COUNT) as u64)];

        let quad_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(quads_buffer)
            .offset(0)
            .range((std::mem::size_of::<primitives::Quad>() * MAX_QUAD_COUNT) as u64)];

        let triangle_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(triangles_buffer)
            .offset(0)
            .range((std::mem::size_of::<primitives::Triangle>() * MAX_TRIANGLE_COUNT) as u64)];

        // Swap radiance image views around each frame
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(radiance_image_views[i])];

        let storage_image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(radiance_image_views[(i + 1) % 2])];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .image_info(&image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .image_info(&storage_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(&buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(&material_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .buffer_info(&bvh_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(5)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .buffer_info(&sphere_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(6)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .buffer_info(&quad_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(7)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .buffer_info(&triangle_buffer_info),
        ];

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    Ok(descriptor_sets)
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    swap_chain_stuff: &SwapChainStuff,
    set_layout: &vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, Vec<vk::Pipeline>)> {
    // let vert_shader_code = read_shader_code(Path::new("shaders/spv/vertex.wgsl.spv"))?;
    let vert_shader_code = read_shader_code(Path::new("shaders/spv/triangle.vert.spv"))?;
    let frag_shader_code = read_shader_code(Path::new("shaders/spv/frag.wgsl.spv"))?;
    // let frag_shader_code = read_shader_code(Path::new("shaders/spv/ray.frag.spv"))?;

    let vert_shader_module = create_shader_module(device, &vert_shader_code)?;
    let frag_shader_module = create_shader_module(device, &frag_shader_code)?;

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(&main_function_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(&main_function_name),
    ];

    // Viewport and Scissor
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let binding_description = [Vertex::get_binding_description()];
    let attribute_descriptions = Vertex::get_attribute_descriptions();

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_descriptions);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewports = [vk::Viewport::default()
        .width(swap_chain_stuff.swapchain_extent.width as f32)
        .height(swap_chain_stuff.swapchain_extent.height as f32)
        .max_depth(1.0)];

    let scissors = [vk::Rect2D::default().extent(swap_chain_stuff.swapchain_extent)];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .viewports(&viewports)
        .scissor_count(1)
        .scissors(&scissors);

    // Rasterizer
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Colour Blending
    let colour_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];

    let colour_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&colour_blend_attachments);

    // let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
    //     .depth_test_enable(true)
    //     .depth_write_enable(true)
    //     .depth_compare_op(vk::CompareOp::LESS);

    let set_layouts = [*set_layout];
    // Pipeline Layout (Uniforms are declared here)
    let format = [swap_chain_stuff.swapchain_format];
    let mut pipeline_rendering_create_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&format);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

    // Actual pipeline
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&colour_blending)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        // .depth_stencil_state(&depth_stencil)
        // .subpass(0)
        .push(&mut pipeline_rendering_create_info)];

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
    };

    unsafe {
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
    }

    match pipeline {
        Ok(v) => Ok((pipeline_layout, v)),
        Err(e) => Err(anyhow!("Failed to create graphics pipeline: {:?}", e)),
    }
}

fn find_memory_type(
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<u32> {
    for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
        if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties)
        {
            return Ok(i as u32);
        }
    }
    return Err(anyhow!("Failed to find a memory type"));
}

fn create_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = match unsafe { device.create_buffer(&buffer_info, None) } {
        Ok(value) => value,
        Err(e) => return Err(anyhow!("Failed to create vertex buffer: {}", e)),
    };

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };
    let memory_type = find_memory_type(
        mem_requirements.memory_type_bits,
        required_memory_properties,
        &mem_properties,
    )?;

    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let memory = unsafe { device.allocate_memory(&allocate_info, None) }?;

    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    Ok((buffer, memory))
}

pub fn copy_buffer(
    device: &ash::Device,
    src: &vk::Buffer,
    dst: &vk::Buffer,
    size: u64,
    command_pool: &vk::CommandPool,
    queue: &vk::Queue,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    unsafe {
        let copy_region = [vk::BufferCopy::default().size(size)];
        device.cmd_copy_buffer(command_buffer, *src, *dst, &copy_region);
    }
    end_single_time_command(device, command_pool, queue, &command_buffer)?;
    Ok(())
}

pub fn create_vertex_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    queue: &vk::Queue,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_size = (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        instance,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    unsafe {
        let data_ptr = device.map_memory(
            staging_buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )? as *mut Vertex;

        data_ptr.copy_from_nonoverlapping(VERTICES.as_ptr(), VERTICES.len());

        device.unmap_memory(staging_buffer_memory);
    };

    let (vertex_buffer, vertex_memory) = create_buffer(
        device,
        instance,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    copy_buffer(
        device,
        &staging_buffer,
        &vertex_buffer,
        buffer_size,
        command_pool,
        queue,
    )?;

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    Ok((vertex_buffer, vertex_memory))
}

pub fn create_index_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    queue: &vk::Queue,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_size = (std::mem::size_of::<Vertex>() * INDICES.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        instance,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    unsafe {
        let data_ptr = device.map_memory(
            staging_buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )? as *mut u16;

        data_ptr.copy_from_nonoverlapping(INDICES.as_ptr(), INDICES.len());

        device.unmap_memory(staging_buffer_memory);
    };

    let (index_buffer, index_memory) = create_buffer(
        device,
        instance,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    copy_buffer(
        device,
        &staging_buffer,
        &index_buffer,
        buffer_size,
        command_pool,
        queue,
    )?;

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    Ok((index_buffer, index_memory))
}

pub fn create_uniform_buffer<T>(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    array_length: u64,
) -> Result<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut T>)> {
    let buffer_size = (std::mem::size_of::<T>() as u64) * array_length;

    let mut uniform_buffers = vec![];
    let mut uniform_buffers_memory = vec![];
    let mut uniform_buffers_mapped = vec![];

    uniform_buffers.reserve_exact(MAX_FRAMES_IN_FLIGHT);
    uniform_buffers_memory.reserve_exact(MAX_FRAMES_IN_FLIGHT);
    uniform_buffers_mapped.reserve_exact(MAX_FRAMES_IN_FLIGHT);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let (uniform_buffer_temp, uniform_buffers_memory_temp) = create_buffer(
            device,
            instance,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        uniform_buffers_mapped.push(unsafe {
            device.map_memory(
                uniform_buffers_memory_temp,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )? as *mut T
        });
        uniform_buffers.push(uniform_buffer_temp);
        uniform_buffers_memory.push(uniform_buffers_memory_temp);
    }
    Ok((
        uniform_buffers,
        uniform_buffers_memory,
        uniform_buffers_mapped,
    ))
}

pub fn create_storage_buffer<T>(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    submit_queue: &vk::Queue,
    data_in: &Vec<T>,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_size = (std::mem::size_of::<T>() * data_in.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        instance,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .context("Failed to map memory for the image texture buffer")?
            as *mut T;

        data_ptr.copy_from_nonoverlapping(data_in.as_ptr(), data_in.len());

        device.unmap_memory(staging_buffer_memory);
    }

    let (buffer, buffer_memory) = create_buffer(
        device,
        instance,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Copy data from staging buffer to our actual one
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    let regions = [vk::BufferCopy::default().size(buffer_size)];
    unsafe { device.cmd_copy_buffer(command_buffer, staging_buffer, buffer, &regions) };
    end_single_time_command(device, command_pool, submit_queue, &command_buffer)?;

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    Ok((buffer, buffer_memory))
}

pub fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
    ];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

    Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
}

pub fn create_command_pool(
    device: &ash::Device,
    queue_family_indices: &QueueFamilyIndices,
) -> Result<vk::CommandPool> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_indices.graphics_family.unwrap());
    Ok(unsafe { device.create_command_pool(&pool_info, None) }?)
}

pub fn create_command_buffers(
    device: &ash::Device,
    command_pool: &vk::CommandPool,
) -> Result<Vec<vk::CommandBuffer>> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }?;
    Ok(command_buffers)
}

pub fn create_sync_object(
    device: &ash::Device,
) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
    // Semaphore is to tell the GPU to wait
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    // Fence is to tell the CPU to wait
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED); // Start the fence signalled so we can immediately wait on it

    let mut semaphores1 = vec![];
    let mut semaphores2 = vec![];
    let mut fences = vec![];

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        unsafe {
            semaphores1.push(device.create_semaphore(&semaphore_info, None)?);
            semaphores2.push(device.create_semaphore(&semaphore_info, None)?);
            fences.push(device.create_fence(&fence_info, None)?);
        }
    }
    Ok((semaphores1, semaphores2, fences))
}

fn create_image(
    device: &ash::Device,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    let texture_image = unsafe {
        device
            .create_image(&image_create_info, None)
            .context("Failed to create Texture Image!")?
    };

    let image_memory_requirement = unsafe { device.get_image_memory_requirements(texture_image) };
    let memory_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(image_memory_requirement.size)
        .memory_type_index(find_memory_type(
            image_memory_requirement.memory_type_bits,
            required_memory_properties,
            device_memory_properties,
        )?);

    let texture_image_memory = unsafe {
        device
            .allocate_memory(&memory_allocate_info, None)
            .context("Failed to allocate texture image memory")?
    };

    unsafe {
        device
            .bind_image_memory(texture_image, texture_image_memory, 0)
            .context("Failed to bind image memory")?
    };

    Ok((texture_image, texture_image_memory))
}

#[allow(unused)]
pub fn create_texture_image(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    submit_queue: &vk::Queue,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let mut image_object = image::ImageReader::open("textures/statue.jpg")?.decode()?; // Apparently this function is slow in debug mode
    image_object = image_object.flipv();
    let (image_width, image_height) = (image_object.width(), image_object.height());
    let image_size =
        (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
    let image_data = image_object.to_rgba8().into_raw();

    if image_size <= 0 {
        return Err(anyhow!("Failed to load texture image"));
    }

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        instance,
        physical_device,
        image_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer_memory,
                0,
                image_size,
                vk::MemoryMapFlags::empty(),
            )
            .context("Failed to map memory for the image texture buffer")?
            as *mut u8;

        data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());

        device.unmap_memory(staging_buffer_memory);
    }

    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };

    let (texture_image, texture_image_memory) = create_image(
        device,
        image_width,
        image_height,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &mem_properties,
    )?;

    transition_image_layout(
        device,
        command_pool,
        submit_queue,
        texture_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;

    copy_buffer_to_image(
        device,
        command_pool,
        submit_queue,
        staging_buffer,
        texture_image,
        image_width,
        image_height,
    )?;

    transition_image_layout(
        device,
        command_pool,
        submit_queue,
        texture_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    Ok((texture_image, texture_image_memory))
}

pub fn create_storage_images(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    submit_queue: &vk::Queue,
) -> Result<([vk::Image; 2], [vk::DeviceMemory; 2], [vk::ImageView; 2])> {
    let image_width = std::cmp::min(
        IDEAL_RADIANCE_IMAGE_SIZE_WIDTH,
        get_max_image_size(instance, physical_device),
    );
    let image_height = std::cmp::min(
        IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT,
        get_max_image_size(instance, physical_device),
    );

    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };

    let mut images = [vk::Image::null(); 2];
    let mut image_memories = [vk::DeviceMemory::null(); 2];
    let mut image_views = [vk::ImageView::null(); 2];

    for i in 0..2 {
        (images[i], image_memories[i]) = create_image(
            device,
            image_width,
            image_height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::STORAGE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &mem_properties,
        )?;

        transition_image_layout(
            device,
            command_pool,
            submit_queue,
            images[i],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        )?;

        image_views[i] = create_image_view(
            device,
            &images[i],
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageAspectFlags::COLOR,
        )?;
    }

    Ok((images, image_memories, image_views))
}

fn begin_single_time_commands(
    device: &ash::Device,
    command_pool: &vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(*command_pool)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)? }[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };
    Ok(command_buffer)
}

fn end_single_time_command(
    device: &ash::Device,
    command_pool: &vk::CommandPool,
    submit_queue: &vk::Queue,
    command_buffer: &vk::CommandBuffer,
) -> Result<()> {
    unsafe { device.end_command_buffer(*command_buffer)? };
    let buffers_to_submit = [*command_buffer];
    let submit_info = [vk::SubmitInfo::default().command_buffers(&buffers_to_submit)];

    unsafe {
        device.queue_submit(*submit_queue, &submit_info, vk::Fence::null())?;
        device.queue_wait_idle(*submit_queue)?;
        device.free_command_buffers(*command_pool, &buffers_to_submit);
    };
    Ok(())
}

pub fn transition_image_layout(
    device: &ash::Device,
    command_pool: &vk::CommandPool,
    submit_queue: &vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let src_access_mask;
    let dst_access_mask;
    let source_stage;
    let destination_stage;

    if old_layout == vk::ImageLayout::UNDEFINED
        && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        destination_stage = vk::PipelineStageFlags::TRANSFER;
    } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        dst_access_mask = vk::AccessFlags::SHADER_READ;
        source_stage = vk::PipelineStageFlags::TRANSFER;
        destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    } else if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::GENERAL {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    } else if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        && new_layout == vk::ImageLayout::PRESENT_SRC_KHR
    {
        src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        dst_access_mask = vk::AccessFlags::empty();
        source_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        destination_stage = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
    } else if old_layout == vk::ImageLayout::UNDEFINED
        && new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        destination_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    } else {
        return Err(anyhow!("Unsupported layout transition!"));
    }

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

    end_single_time_command(device, command_pool, submit_queue, &command_buffer)?;
    Ok(())
}

fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: &vk::CommandPool,
    submit_queue: &vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let regions = [vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })];

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        )
    };

    end_single_time_command(device, command_pool, submit_queue, &command_buffer)?;
    Ok(())
}

#[allow(unused)]
pub fn create_texture_image_view(
    device: &ash::Device,
    texture_image: &vk::Image,
) -> Result<vk::ImageView> {
    Ok(create_image_view(
        device,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
    )?)
}

#[allow(unused)]
pub fn create_texture_sampler(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> Result<vk::Sampler> {
    let device_properties = unsafe { instance.get_physical_device_properties(*physical_device) };
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

fn find_supported_format(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    candidates: &Vec<vk::Format>,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    for &format in candidates {
        let props =
            unsafe { instance.get_physical_device_format_properties(*physical_device, format) };

        if tiling == vk::ImageTiling::LINEAR
            && (props.linear_tiling_features & features) == features
        {
            return Ok(format);
        } else if tiling == vk::ImageTiling::OPTIMAL
            && (props.optimal_tiling_features & features) == features
        {
            return Ok(format);
        }
    }
    Err(anyhow!("Failed to find a supported format"))
}

fn find_depth_format(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> Result<vk::Format> {
    let candidates = vec![
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    Ok(find_supported_format(
        instance,
        physical_device,
        &candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )?)
}

// fn has_stencil_component(format: vk::Format) -> bool {
//     format == vk::Format::D24_UNORM_S8_UINT || format == vk::Format::D32_SFLOAT_S8_UINT
// }

pub fn create_depth_resources(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    swap_chain_stuff: &SwapChainStuff,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let depth_format = find_depth_format(instance, physical_device)?;
    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };
    let (image, image_memory) = create_image(
        device,
        swap_chain_stuff.swapchain_extent.width,
        swap_chain_stuff.swapchain_extent.height,
        depth_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &mem_properties,
    )?;
    let image_view = create_image_view(device, &image, depth_format, vk::ImageAspectFlags::DEPTH)?;

    Ok((image, image_memory, image_view))
}

pub fn prepare_timestamp_queries(device: &ash::Device) -> Result<(vk::QueryPool, Vec<u64>)> {
    let timestamps = vec![0; 2];
    let query_pool_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(timestamps.len() as u32);

    let timestamps_query_pool = unsafe { device.create_query_pool(&query_pool_info, None)? };

    Ok((timestamps_query_pool, timestamps))
}
