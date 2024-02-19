use ash::{prelude::VkResult, vk};

use crate::{context::DeviceContext, resource::Image};

pub trait Output<'ctx> {
    fn image_view(&self) -> vk::ImageView;
}

pub struct OutputImage<'ctx> {
    image: Image<'ctx>,
    view: vk::ImageView,
}

impl<'ctx> Output<'ctx> for OutputImage<'ctx> {
    fn image_view(&self) -> vk::ImageView {
        self.view
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

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UINT)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = context.device().create_image_view(&view_info, None)?;
        
        Ok(Self {
            image,
            view,
        })
    } 
}