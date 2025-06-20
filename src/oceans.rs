use std::{array, ffi::CString, path::Path, sync::Arc, time::Instant};

use anyhow::{anyhow, Result};
use ash::vk;

use crate::{
    core::{self, begin_single_time_commands, create_shader_module, end_single_time_command},
    primitives::{self},
    structures,
    tools::read_shader_code,
    vulkan::{Destructor, Image},
    MAX_FRAMES_IN_FLIGHT,
};

const FFT_IMAGES: usize = 1; // How many images do we need?
pub const OCEAN_RESOLUTION: u32 = 1024;

enum PipelineType {
    SpectraGeneration,
    FFTx,
    FFTy,
}

#[derive(Debug, Default)]
#[repr(C)]
struct PushConstants {
    pub wind_speed: f32,    // At 10m in m/s
    pub wind_angle: f32,    // Radians from north
    pub leeward_fetch: f32, // Distance in m from nearest shore downwind
    pub depth: f32,         // Depth in m
    pub length_scale: f32,
    pub size: u32,
    pub timestamp_ns: u64, // Nanoseconds from start of running
}

impl PushConstants {
    pub fn as_slice(&self) -> &[u8; size_of::<Self>()] {
        unsafe { &*(self as *const Self as *const [u8; size_of::<Self>()]) }
    }
}

pub struct Ocean<'a> {
    start_time: Instant,
    pub images: [Image<'a>; FFT_IMAGES * MAX_FRAMES_IN_FLIGHT],

    entity: Option<hecs::Entity>,
    push_constants: PushConstants,

    _spectra_descriptor_pool: Destructor<vk::DescriptorPool>,
    _spectra_descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    spectra_descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    spectra_pipeline_layout: Destructor<vk::PipelineLayout>,
    spectra_pipeline: Destructor<vk::Pipeline>,
    spectra_commands: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    _spectra_command_pool: Destructor<vk::CommandPool>,

    _fftx_descriptor_pool: Destructor<vk::DescriptorPool>,
    _fftx_descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    _fftx_descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    _fftx_pipeline_layout: Destructor<vk::PipelineLayout>,
    _fftx_pipeline: Destructor<vk::Pipeline>,
    fftx_commands: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    _fftx_command_pool: Destructor<vk::CommandPool>,

    _ffty_descriptor_pool: Destructor<vk::DescriptorPool>,
    _ffty_descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    _ffty_descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    _ffty_pipeline_layout: Destructor<vk::PipelineLayout>,
    _ffty_pipeline: Destructor<vk::Pipeline>,
    ffty_commands: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    _ffty_command_pool: Destructor<vk::CommandPool>,

    semaphore_values: [u64; MAX_FRAMES_IN_FLIGHT],

    #[allow(dead_code)]
    queue: vk::Queue,
    semaphores: [Destructor<vk::Semaphore>; MAX_FRAMES_IN_FLIGHT],
}

impl<'a> Ocean<'a> {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<vk_mem::Allocator>,
        queue_family_indices: structures::QueueFamilyIndices,
    ) -> Result<Self> {
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());
        let (spectra_command_pool, spectra_commands) = core::create_commands_flight_frames(
            &device,
            queue_family_indices.compute_family.unwrap().0,
        )?;
        let (fftx_command_pool, fftx_commands) = core::create_commands_flight_frames(
            &device,
            queue_family_indices.compute_family.unwrap().0,
        )?;
        let (ffty_command_pool, ffty_commands) = core::create_commands_flight_frames(
            &device,
            queue_family_indices.compute_family.unwrap().0,
        )?;

        let images = create_images(
            device,
            allocator,
            OCEAN_RESOLUTION,
            spectra_command_pool.get(),
            queue,
        )?;

        log::trace!(
            "FFT has images {:?} and {:?}",
            images[0].get(),
            images[1].get()
        );

        let (
            spectra_descriptor_pool,
            spectra_descriptor_set_layout,
            spectra_descriptor_sets,
            spectra_pipeline_layout,
            spectra_pipeline,
        ) = create_pipeline(device, &images, PipelineType::SpectraGeneration)?;
        let (
            fftx_descriptor_pool,
            fftx_descriptor_set_layout,
            fftx_descriptor_sets,
            fftx_pipeline_layout,
            fftx_pipeline,
        ) = create_pipeline(device, &images, PipelineType::FFTx)?;
        let (
            ffty_descriptor_pool,
            ffty_descriptor_set_layout,
            ffty_descriptor_sets,
            ffty_pipeline_layout,
            ffty_pipeline,
        ) = create_pipeline(device, &images, PipelineType::FFTy)?;

        create_fft_commands(
            device,
            &fftx_commands,
            &images,
            fftx_pipeline.get(),
            fftx_pipeline_layout.get(),
            &fftx_descriptor_sets,
        )?;

        create_fft_commands(
            device,
            &ffty_commands,
            &images,
            ffty_pipeline.get(),
            ffty_pipeline_layout.get(),
            &ffty_descriptor_sets,
        )?;

        let semaphores = array::from_fn(|_| {
            Destructor::new(
                &device,
                core::create_semaphore(&device).unwrap(),
                device.fp_v1_0().destroy_semaphore,
            )
        });

        Ok(Self {
            start_time: Instant::now(),
            images,
            entity: None,
            push_constants: PushConstants::default(),
            _spectra_descriptor_set_layout: spectra_descriptor_set_layout,
            spectra_pipeline_layout,
            spectra_pipeline,
            _spectra_descriptor_pool: spectra_descriptor_pool,
            spectra_descriptor_sets,
            _spectra_command_pool: spectra_command_pool,
            spectra_commands,

            _fftx_descriptor_set_layout: fftx_descriptor_set_layout,
            _fftx_pipeline_layout: fftx_pipeline_layout,
            _fftx_pipeline: fftx_pipeline,
            _fftx_descriptor_pool: fftx_descriptor_pool,
            _fftx_descriptor_sets: fftx_descriptor_sets,
            _fftx_command_pool: fftx_command_pool,
            fftx_commands,

            _ffty_descriptor_set_layout: ffty_descriptor_set_layout,
            _ffty_pipeline_layout: ffty_pipeline_layout,
            _ffty_pipeline: ffty_pipeline,
            _ffty_descriptor_pool: ffty_descriptor_pool,
            _ffty_descriptor_sets: ffty_descriptor_sets,
            _ffty_command_pool: ffty_command_pool,
            ffty_commands,

            queue,
            semaphores,
            semaphore_values: [0; 2],
        })
    }
    pub fn update(&mut self, entity: hecs::Entity, ocean: &primitives::ocean::Ocean) -> bool {
        match self.entity {
            None => self.entity = Some(entity),
            Some(e) if e != entity => {
                log::warn!("Multiple ocean primitives declared, ignoring {:?}", e);
                return false;
            }
            _ => {}
        }
        self.push_constants.wind_speed = ocean.params.wind_speed;
        self.push_constants.wind_angle = ocean.params.wind_angle;
        self.push_constants.leeward_fetch = ocean.params.leeward_fetch;
        self.push_constants.depth = ocean.params.depth;
        self.push_constants.length_scale = ocean.params.length_scale;
        self.push_constants.size = ocean.params.size;
        return true;
    }
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        frame_index: usize,
    ) -> Result<(vk::Semaphore, u64)> {
        // If we haven't initialised an ocean we don't want to run the compute shaders
        if let None = self.entity {
            return Ok((self.semaphores[frame_index].get(), 0));
        }

        // Update Push constants
        self.push_constants.timestamp_ns = self.start_time.elapsed().as_nanos() as u64;
        // Generate the ocean wave spectra in the images
        self.dispatch_spectra_generation(device, frame_index)?;
        self.semaphore_values[frame_index] += 1;
        // Convert the ocean wave spectra to a height and normal map
        self.dispatch_fft(device, frame_index, PipelineType::FFTx)?;
        self.semaphore_values[frame_index] += 1;
        self.dispatch_fft(device, frame_index, PipelineType::FFTy)?;
        self.semaphore_values[frame_index] += 1;
        Ok((
            self.semaphores[frame_index].get(),
            self.semaphore_values[frame_index],
        ))
    }
    fn dispatch_fft(
        &mut self,
        device: &ash::Device,
        frame_index: usize,
        fft_type: PipelineType,
    ) -> Result<()> {
        let command_buffer_infos = [vk::CommandBufferSubmitInfo {
            command_buffer: match fft_type {
                PipelineType::FFTx => self.fftx_commands[frame_index],
                PipelineType::FFTy => self.ffty_commands[frame_index],
                _ => {
                    log::warn!("Tried to dispatch FFT with wrong type of pipeline");
                    return Ok(());
                }
            },
            ..Default::default()
        }];
        // Wait for the spectra generation to finish
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo {
            semaphore: self.semaphores[frame_index].get(),
            value: self.semaphore_values[frame_index],
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            ..Default::default()
        }];
        // Let Ray know that we are finished
        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo {
            semaphore: self.semaphores[frame_index].get(),
            value: self.semaphore_values[frame_index] + 1,
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER, // TODO: Should this be Ray Tracing?
            ..Default::default()
        }];
        let submits = [vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)];

        unsafe {
            device.queue_submit2(self.queue, &submits, vk::Fence::null())?;
        };
        Ok(())
    }
    fn dispatch_spectra_generation(
        &mut self,
        device: &ash::Device,
        frame_index: usize,
    ) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(self.spectra_commands[frame_index], &begin_info)? };
        // Convert Images from Ray to be able to write to them
        convert_images(
            device,
            self.spectra_commands[frame_index],
            &self.images[frame_index * FFT_IMAGES..(frame_index + 1) * FFT_IMAGES],
            false,
        );

        unsafe {
            // Update the push constants
            device.cmd_push_constants(
                self.spectra_commands[frame_index],
                self.spectra_pipeline_layout.get(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                self.push_constants.as_slice(),
            );
            // Set up the spectra generation shader
            device.cmd_bind_pipeline(
                self.spectra_commands[frame_index],
                vk::PipelineBindPoint::COMPUTE,
                self.spectra_pipeline.get(),
            );
            let descriptor_sets_to_bind = [self.spectra_descriptor_sets[frame_index]];
            device.cmd_bind_descriptor_sets(
                self.spectra_commands[frame_index],
                vk::PipelineBindPoint::COMPUTE,
                self.spectra_pipeline_layout.get(),
                0,
                &descriptor_sets_to_bind,
                &[],
            );
            // Dispatch the generation of the ocean wave spectra
            device.cmd_dispatch(
                self.spectra_commands[frame_index],
                OCEAN_RESOLUTION / 16,
                OCEAN_RESOLUTION / 16,
                1,
            );
            device.end_command_buffer(self.spectra_commands[frame_index])?;
        }

        let command_buffer_infos = [vk::CommandBufferSubmitInfo {
            command_buffer: self.spectra_commands[frame_index],
            ..Default::default()
        }];
        // Wait for the last run to finish
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo {
            semaphore: self.semaphores[frame_index].get(),
            value: self.semaphore_values[frame_index],
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            ..Default::default()
        }];
        // Let FFT know that we are finished
        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo {
            semaphore: self.semaphores[frame_index].get(),
            value: self.semaphore_values[frame_index] + 1,
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            ..Default::default()
        }];
        let submits = [vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)];

        unsafe {
            device.queue_submit2(self.queue, &submits, vk::Fence::null())?;
        };
        Ok(())
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
    pipeline_type: PipelineType,
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

    let shader_code = match pipeline_type {
        PipelineType::SpectraGeneration => {
            read_shader_code(Path::new("shaders/spv/ocean.slang.spv"))?
        }
        PipelineType::FFTx | PipelineType::FFTy => {
            read_shader_code(Path::new("shaders/spv/fft.slang.spv"))?
        }
    };
    let shader_module = create_shader_module(&device, &shader_code)?;
    log::info!("Processing shader {:?}", shader_module.get());
    let main_function_name = match pipeline_type {
        PipelineType::SpectraGeneration => CString::new("main").unwrap(),
        PipelineType::FFTx => CString::new("fftx").unwrap(),
        PipelineType::FFTy => CString::new("ffty").unwrap(),
    };
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

fn create_fft_commands(
    device: &ash::Device,
    command_buffers: &[vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    images: &[Image; FFT_IMAGES * MAX_FRAMES_IN_FLIGHT],
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_sets: &[vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
) -> Result<()> {
    for frame_index in 0..MAX_FRAMES_IN_FLIGHT {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(command_buffers[frame_index], &begin_info)? };
        // Convert Images from Ray to be able to write to them
        convert_images(
            device,
            command_buffers[frame_index],
            &images[frame_index * FFT_IMAGES..(frame_index + 1) * FFT_IMAGES],
            false,
        );

        unsafe {
            // Dispatch the compute shader
            device.cmd_bind_pipeline(
                command_buffers[frame_index],
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            let descriptor_sets_to_bind = [descriptor_sets[frame_index]];
            device.cmd_bind_descriptor_sets(
                command_buffers[frame_index],
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );
            device.cmd_dispatch(command_buffers[frame_index], OCEAN_RESOLUTION, 1, 1);
            // Convert Images for Ray to be able to read them
            convert_images(
                device,
                command_buffers[frame_index],
                &images[frame_index * FFT_IMAGES..(frame_index + 1) * FFT_IMAGES],
                true,
            );
            device.end_command_buffer(command_buffers[frame_index])?;
        }
    }
    Ok(())
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
