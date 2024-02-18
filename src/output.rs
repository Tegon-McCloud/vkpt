use ash::vk;

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
    pub fn new(context: &'ctx DeviceContext, resolution: (u32, u32)) {
        
    } 
}