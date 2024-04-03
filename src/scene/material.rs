use crate::pipeline::Shader;

use super::MaterialTypeHandle;

pub struct MaterialType<'ctx> {
    pub evaluation_shader: Shader<'ctx>,
    pub sample_shader: Shader<'ctx>,
}

pub struct Material {
    pub ior: f32,
    pub roughness: f32,
    pub material_type: MaterialTypeHandle,
}
