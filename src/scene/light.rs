use std::ffi::CStr;

use ash::prelude::VkResult;

use crate::{context::DeviceContext, pipeline::{Shader, ShaderResourceLayout}, resource::Image};

pub trait LightSource<'ctx> {
    // fn sample_shader() -> Shader<'ctx>;
    fn miss_shader(&self) -> Option<&Shader<'ctx>>;
}


pub struct Environment<'ctx> {
    miss_shader: Shader<'ctx>,
    image: Option<Image<'ctx>>,
}

impl<'ctx> Environment<'ctx> {
    pub fn constant(context: &'ctx DeviceContext) -> VkResult<Environment> {

        let miss_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/constant.rmiss.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
                ShaderResourceLayout::default(),
            )?
        };

        Ok(Environment {
            miss_shader,
            image: None,
        })
    }


    pub fn spherical(context: &'ctx DeviceContext, image: Image<'ctx>) -> VkResult<Environment<'ctx>> {

        let miss_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/spherical.rmiss.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
                ShaderResourceLayout::new(vec![], 0),
            )?
        };

        Ok(Environment {
            miss_shader,
            image: Some(image),
        })
    }
}

impl<'ctx> LightSource<'ctx> for Environment<'ctx> {
    fn miss_shader(&self) -> Option<&Shader<'ctx>> {
        Some(&self.miss_shader)
    }
}
