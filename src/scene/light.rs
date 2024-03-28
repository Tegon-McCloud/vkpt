use std::ffi::CStr;

use ash::prelude::VkResult;

use crate::{context::DeviceContext, pipeline::Shader};

use super::TextureHandle;

pub trait LightSource<'ctx> {
    // fn sample_shader() -> Shader<'ctx>;
    fn miss_shader(&self) -> Option<&Shader<'ctx>>;
}


pub struct Environment<'ctx> {
    miss_shader: Shader<'ctx>,
    texture: Option<TextureHandle>,
}

impl<'ctx> Environment<'ctx> {
    pub fn constant(context: &'ctx DeviceContext) -> VkResult<Environment> {

        let miss_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/constant.rmiss.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
            )?
        };

        Ok(Environment {
            miss_shader,
            texture: None,
        })
    }


    pub fn spherical(context: &'ctx DeviceContext, texture: TextureHandle) -> VkResult<Environment<'ctx>> {

        let miss_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/spherical.rmiss.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
            )?
        };

        Ok(Environment {
            miss_shader,
            texture: Some(texture),
        })
    }
}

impl<'ctx> LightSource<'ctx> for Environment<'ctx> {
    fn miss_shader(&self) -> Option<&Shader<'ctx>> {
        Some(&self.miss_shader)
    }
}
