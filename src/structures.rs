use anyhow::{Context, Result};
use ash::vk;

use crate::{
    vulkan::{self, Destructor},
    WINDOW_HEIGHT, WINDOW_WIDTH,
};

pub struct SurfaceStuff {
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
}

impl SurfaceStuff {
    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> Result<Self> {
        use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        Ok(Self {
            surface_loader,
            surface,
        })
    }
}

impl Drop for SurfaceStuff {
    fn drop(&mut self) {
        log::trace!("Dropping Surface");
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

pub struct SwapChainStuff {
    pub image_views: Vec<Destructor<vk::ImageView>>,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub swapchain: vulkan::Swapchain,
}

impl SwapChainStuff {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_stuff: &SurfaceStuff,
        queue_family: &QueueFamilyIndices,
    ) -> Result<Self> {
        let swapchain_support = SwapChainSupportDetails::query(physical_device, surface_stuff);

        let surface_format = swapchain_support.choose_swap_surface_format();
        let present_mode = swapchain_support.choose_swap_present_mode();
        let extent = swapchain_support.choose_swap_extent();

        // Decide how many images we want in our swap chain
        // (Recommended to request at least one more than the minimum to avoid having to wait for the driver)
        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_stuff.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1) // Number of layers in each image (always 1 unless doing stereoscopic 3D)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) // Declare the type of operations that are going to be performed on the images in the swap chain (COLOR_ATTACHMENT for rendering directly)
            .present_mode(present_mode)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // How the window should deal with alpha components in pixels
            .clipped(true); // Ignore pixels that are obscured(i.e. hidden behind other windows)

        let queue_family_total = vec![
            queue_family.graphics_family.unwrap().0,
            queue_family.present_family.unwrap().0,
        ];
        // Check if we are just using one queue for everything
        if queue_family.graphics_family != queue_family.present_family {
            swapchain_create_info = swapchain_create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_total[..]);
        }

        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
        let swapchain = vulkan::Swapchain::new(swapchain_loader, swapchain_create_info)
            .context("Unable to create swapchain")?;

        let swapchain_raw_images = unsafe {
            swapchain
                .loader
                .get_swapchain_images(swapchain.get())
                .context("Failed to get Swapchain Images")?
        };

        let image_views = create_image_views(device, surface_format.format, &swapchain_raw_images)?;

        Ok(SwapChainStuff {
            swapchain,
            format: surface_format.format,
            extent,
            images: swapchain_raw_images,
            image_views,
        })
    }
    pub fn reset(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_stuff: &SurfaceStuff,
        queue_family: &QueueFamilyIndices,
    ) -> Result<()> {
        // Drop old swapchain
        for i in &mut self.image_views {
            i.empty();
        }
        self.swapchain.empty();
        *self = SwapChainStuff::new(
            instance,
            device,
            physical_device,
            surface_stuff,
            queue_family,
        )?;
        Ok(())
    }
    pub fn get_swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain.get()
    }
    pub fn get_loader(&self) -> &ash::khr::swapchain::Device {
        &self.swapchain.loader
    }
}

#[derive(Debug, Clone, Copy)]
// Stores the family and index of each each queue that we want
pub struct QueueFamilyIndices {
    pub graphics_family: Option<(u32, u32)>,
    pub present_family: Option<(u32, u32)>,
    pub compute_family: Option<(u32, u32)>,
    pub transfer_family: Option<(u32, u32)>,
}

impl QueueFamilyIndices {
    pub fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
            compute_family: None,
            transfer_family: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
            && self.present_family.is_some()
            && self.compute_family.is_some()
    }

    pub fn is_index_taken(&self, queue_index: u32, index: u32) -> bool {
        if self
            .graphics_family
            .is_some_and(|x| x.0 == queue_index && x.1 == index)
            || self
                .compute_family
                .is_some_and(|x| x.0 == queue_index && x.1 == index)
            || self
                .transfer_family
                .is_some_and(|x| x.0 == queue_index && x.1 == index)
            || self
                .present_family
                .is_some_and(|x| x.0 == queue_index && x.1 == index)
        {
            true
        } else {
            false
        }
    }
}

pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    pub fn query(physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> Self {
        unsafe {
            let capabilities = surface_stuff
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface_stuff.surface)
                .expect("Failed to query for surface capabilities.");
            let formats = surface_stuff
                .surface_loader
                .get_physical_device_surface_formats(physical_device, surface_stuff.surface)
                .expect("Failed to query for surface formats.");
            let present_modes = surface_stuff
                .surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface_stuff.surface)
                .expect("Failed to query for surface present mode.");

            SwapChainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    /// Select the resolution of images in the swap chain
    pub fn choose_swap_extent(&self) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::max_value() {
            self.capabilities.current_extent
        } else {
            use num::clamp;

            vk::Extent2D {
                width: clamp(
                    WINDOW_WIDTH,
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                ),
                height: clamp(
                    WINDOW_HEIGHT,
                    self.capabilities.min_image_extent.height,
                    self.capabilities.max_image_extent.height,
                ),
            }
        }
    }

    /// Select the colour depth from the ones available
    pub fn choose_swap_surface_format(&self) -> vk::SurfaceFormatKHR {
        for available_format in &self.formats {
            if available_format.format == vk::Format::B8G8R8A8_SRGB
                && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *available_format;
            }
        }
        self.formats[0]
    }

    /// Select the conditions for 'swapping' images to the screen
    pub fn choose_swap_present_mode(&self) -> vk::PresentModeKHR {
        // for available_mode in &self.present_modes {
        //     // Mailbox is good if energy efficiency is not an issue (mobile should probably use FIFO)
        //     if *available_mode == vk::PresentModeKHR::MAILBOX {
        //         return *available_mode;
        //     }
        // }
        vk::PresentModeKHR::FIFO // The only mode that is guaranteed to be available
    }
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
) -> Result<Destructor<vk::ImageView>> {
    let create_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect_flags,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    Ok(Destructor::new(
        device,
        unsafe { device.create_image_view(&create_info, None)? },
        device.fp_v1_0().destroy_image_view,
    ))
}

fn create_image_views(
    device: &ash::Device,
    surface_format: vk::Format,
    swapchain_images: &Vec<vk::Image>,
) -> Result<Vec<Destructor<vk::ImageView>>> {
    let mut image_views = vec![];
    for &image in swapchain_images {
        image_views.push(create_image_view(
            device,
            image,
            surface_format,
            vk::ImageAspectFlags::COLOR,
        )?);
    }
    Ok(image_views)
}
