use ash::{prelude::VkResult, vk};

use crate::{context::DeviceContext, resource::{Image, ImageView}};

pub trait Output<'ctx> {
    unsafe fn image_view(&self) -> vk::ImageView;
}

pub struct OutputImage<'ctx> {
    #[allow(unused)]
    image: Image<'ctx>,
    view: ImageView<'ctx>,
}

impl<'ctx> Output<'ctx> for OutputImage<'ctx> {
    unsafe fn image_view(&self) -> vk::ImageView {
        self.view.inner()
    }
}

impl<'ctx> OutputImage<'ctx> {
    pub unsafe fn new(context: &'ctx DeviceContext, resolution: (u32, u32)) -> VkResult<Self> {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UINT)
            .extent(vk::Extent3D { width: resolution.0, height: resolution.1, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST);

        let image = Image::new(context, &image_info, gpu_allocator::MemoryLocation::GpuOnly)?;

        let view = ImageView::new(&image, vk::Format::R8G8B8A8_UINT, 0..1, 0..1)?;
            
        Ok(Self {
            image,
            view,
        })
    } 
}