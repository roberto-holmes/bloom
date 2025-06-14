use std::{ffi::CString, path::Path, sync::Arc, time::Instant};

use anyhow::{anyhow, Result};
use ash::vk;

use crate::{
    core::{self, begin_single_time_commands, create_shader_module, end_single_time_command},
    structures,
    tools::read_shader_code,
    vulkan::{Destructor, Image},
    MAX_FRAMES_IN_FLIGHT,
};

const FFT_IMAGES: usize = 1; // How many images do we need?
const OCEAN_RESOLUTION: u32 = 1024;

#[derive(Debug, Default)]
#[repr(C)]
struct PushConstants {
    pub wind_speed_x: f32, // At 10m in m/s
    pub wind_speed_z: f32,
    pub wind_fetch_x: f32, // Distannce from shore in m
    pub wind_fetch_z: f32,
    pub timestamp_ns: u64, // Nanoseconds from start of running
}

impl PushConstants {
    pub fn as_slice(&self) -> &[u8; size_of::<PushConstants>()] {
        unsafe { &*(self as *const Self as *const [u8; size_of::<PushConstants>()]) }
    }
}

pub struct Ocean<'a> {
    start_time: Instant,
    pub images: [Image<'a>; FFT_IMAGES * MAX_FRAMES_IN_FLIGHT],

    push_constants: PushConstants,

    _descriptor_pool: Destructor<vk::DescriptorPool>,
    _descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
}

impl<'a> Ocean<'a> {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<vk_mem::Allocator>,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {
        let images = create_images(device, allocator, OCEAN_RESOLUTION, command_pool, queue)?;

        log::trace!(
            "FFT has images {:?} and {:?}",
            images[0].get(),
            images[1].get()
        );

        let (descriptor_pool, descriptor_set_layout, descriptor_sets, pipeline_layout, pipeline) =
            create_pipeline(device, &images)?;

        Ok(Self {
            start_time: Instant::now(),
            images,
            push_constants: PushConstants::default(),
            _descriptor_set_layout: descriptor_set_layout,
            pipeline_layout,
            pipeline,
            _descriptor_pool: descriptor_pool,
            descriptor_sets,
        })
    }
    pub fn update(&mut self) {
        self.push_constants.timestamp_ns = self.start_time.elapsed().as_nanos() as u64;
    }
    pub fn dispatch(
        &self,
        device: &ash::Device,
        frame_index: usize,
        command_buffer: vk::CommandBuffer,
    ) {
        // Convert Images from Ray to be able to write to them
        convert_images(
            device,
            command_buffer,
            &self.images[frame_index * FFT_IMAGES..(frame_index + 1) * FFT_IMAGES],
            false,
        );

        unsafe {
            // Update the push constants
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout.get(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                self.push_constants.as_slice(),
            );

            // Dispatch the compute shader
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.get(),
            );
            let descriptor_sets_to_bind = [self.descriptor_sets[frame_index]];
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout.get(),
                0,
                &descriptor_sets_to_bind,
                &[],
            );
            // build_barrier(device, command_buffer, true);
            device.cmd_dispatch(command_buffer, 8, 8, 1); // TODO: Sizes

            // Convert Images for Ray to be able to read them
            convert_images(
                device,
                command_buffer,
                &self.images[frame_index * FFT_IMAGES..(frame_index + 1) * FFT_IMAGES],
                true,
            );
        }
    }
}

fn create_images<'a>(
    device: &ash::Device,
    allocator: &Arc<vk_mem::Allocator>,
    size: u32,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
) -> Result<[Image<'a>; FFT_IMAGES * MAX_FRAMES_IN_FLIGHT]> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    let images: [Image<'a>; FFT_IMAGES * MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| {
        // Create the image in memory
        let i = Image::new(
            device,
            allocator,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
            size,
            size,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )
        .unwrap();

        // Convert the image to the format we need
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(i.get())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)];

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        i
    });
    end_single_time_command(device, command_pool, submit_queue, command_buffer)?;
    Ok(images)
}

fn create_pipeline(
    device: &ash::Device,
    output_images: &[Image; FFT_IMAGES * MAX_FRAMES_IN_FLIGHT],
) -> Result<(
    Destructor<vk::DescriptorPool>,
    Destructor<vk::DescriptorSetLayout>,
    [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    Destructor<vk::PipelineLayout>,
    Destructor<vk::Pipeline>,
)> {
    let set_layout_bindings: [vk::DescriptorSetLayoutBinding<'_>; FFT_IMAGES] =
        std::array::from_fn(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
    let descriptor_layout =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_layout_bindings);
    let descriptor_set_layout = Destructor::new(
        device,
        unsafe { device.create_descriptor_set_layout(&descriptor_layout, None)? },
        device.fp_v1_0().destroy_descriptor_set_layout,
    );
    let set_layouts: [vk::DescriptorSetLayout; MAX_FRAMES_IN_FLIGHT] =
        std::array::from_fn(|_| descriptor_set_layout.get());

    let push_constants = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        size: size_of::<PushConstants>() as u32,
        offset: 0,
    }];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constants);
    let pipeline_layout = Destructor::new(
        device,
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? },
        device.fp_v1_0().destroy_pipeline_layout,
    );

    let pool_sizes: [vk::DescriptorPoolSize; FFT_IMAGES] = std::array::from_fn(|_| {
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE)
    });

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

    let descriptor_pool = Destructor::new(
        device,
        unsafe { device.create_descriptor_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_descriptor_pool,
    );

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool.get())
        .set_layouts(set_layouts.as_slice());
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        // Swap image views around each frame
        let image_infos: [[vk::DescriptorImageInfo; 1]; FFT_IMAGES] = std::array::from_fn(|j| {
            [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(output_images[FFT_IMAGES * i + j].view())]
        });
        let descriptor_writes: [vk::WriteDescriptorSet<'_>; FFT_IMAGES] =
            std::array::from_fn(|j| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(j as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .image_info(&image_infos[j])
            });
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    let shader_code = read_shader_code(Path::new("shaders/spv/fft.slang.spv"))?;
    let shader_module = create_shader_module(&device, &shader_code)?;
    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module.get())
        .name(&main_function_name);

    let pipeline_create_infos = [vk::ComputePipelineCreateInfo::default()
        .layout(pipeline_layout.get())
        .stage(stage)];
    let pipeline = Destructor::new(
        device,
        match unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
        } {
            Ok(v) => v[0], // We are only creating one timeline so we only want the first object in the vector
            Err(e) => return Err(anyhow!(e.1)),
        },
        device.fp_v1_0().destroy_pipeline,
    );

    let descriptor_sets = <Vec<vk::DescriptorSet> as TryInto<
        [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    >>::try_into(descriptor_sets)
    .unwrap();

    Ok((
        descriptor_pool,
        descriptor_set_layout,
        descriptor_sets,
        pipeline_layout,
        pipeline,
    ))
}

fn convert_images(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    images: &[Image],
    to_ray: bool,
) {
    let barriers: [vk::ImageMemoryBarrier2; FFT_IMAGES] = std::array::from_fn(|i| {
        vk::ImageMemoryBarrier2::default()
            // .old_layout(vk::ImageLayout::UNDEFINED)
            // .new_layout(vk::ImageLayout::GENERAL)
            .old_layout(if to_ray {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(if to_ray {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::GENERAL
            })
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(images[i].get())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_stage_mask(if to_ray {
                vk::PipelineStageFlags2::COMPUTE_SHADER
                // vk::PipelineStageFlags2::ALL_COMMANDS
            } else {
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                // vk::PipelineStageFlags2::ALL_COMMANDS
            })
            .dst_stage_mask(if to_ray {
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                // vk::PipelineStageFlags2::ALL_COMMANDS
            } else {
                vk::PipelineStageFlags2::COMPUTE_SHADER
                // vk::PipelineStageFlags2::ALL_COMMANDS
            })
            .src_access_mask(if to_ray {
                vk::AccessFlags2::SHADER_WRITE
            } else {
                vk::AccessFlags2::SHADER_READ
            })
            .dst_access_mask(if to_ray {
                vk::AccessFlags2::SHADER_READ
            } else {
                vk::AccessFlags2::SHADER_WRITE
            })
    });

    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);

    unsafe {
        device.cmd_pipeline_barrier2(command_buffer, &dependency);
    }
}
