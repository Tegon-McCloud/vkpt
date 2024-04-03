use std::{ffi::{CStr, CString}, fs::File, io::{Read, Seek, SeekFrom}, path::Path};

use ash::{prelude::VkResult, vk};

use crate::{context::DeviceContext, shader_binding_table::{ShaderBindingTable, ShaderBindingTableDescription}, util};

pub trait ShaderData {
    fn as_u8_slice<'a>(&'a self) -> &'a [u8];
}

pub struct Shader<'a> {
    context: &'a DeviceContext,
    module: vk::ShaderModule,
    entry_point: CString,
}

impl<'a> Shader<'a> {
    pub unsafe fn new<P: AsRef<Path>>(
        context: &'a DeviceContext,
        spv_path: P,
        entry_point: CString,
    ) -> VkResult<Self> {
        let spv_code = Self::read_spv(spv_path.as_ref());
        let info = vk::ShaderModuleCreateInfo::builder()
            .code(&spv_code);
        let module = context.create_shader_module(&info)?;
        
        Ok(Self {
            context,
            module,
            entry_point,
        })
    }

    fn read_spv(path: &Path) -> Vec<u32> {
        let mut file = File::open(path).expect("failed to open SPIR-V file");
        
        let spv_size = file.seek(SeekFrom::End(0)).expect("failed to seek in SPIR-V file") as usize;
        file.seek(SeekFrom::Start(0)).expect("failed to seek in SPIR-V file");
        
        let u32_buffer_size = spv_size.div_ceil(std::mem::size_of::<u32>());
    
        let mut spv_code = vec![0u32; u32_buffer_size];
        
        unsafe {
            file.read(util::slice_as_mut_u8_slice(spv_code.as_mut_slice())).expect("failed to read SPIR-V file");
        }

        spv_code
    }

    pub unsafe fn module(&self) -> vk::ShaderModule {
        self.module
    }

    pub fn entry_point(&self) -> &CStr {
        &self.entry_point
    }

}


impl<'a> Drop for Shader<'a> {
    fn drop(&mut self) {
        unsafe {
            self.context.destroy_shader_module(self.module);
        }
    }
}

#[derive(Clone, Copy)]
pub enum ShaderGroup<'ctx, 'a> {
    Raygen {
        raygen: &'a Shader<'ctx>,
    },

    Miss {
        miss: &'a Shader<'ctx>,
    },

    TriangleHit {
        closest_hit: &'a Shader<'ctx>,
    },

    Callable {
        callable: &'a Shader<'ctx>,
    },
}

pub struct ResourceLayout {
    sets: Vec<vk::DescriptorSetLayout>,
    constants: Vec<vk::PushConstantRange>,
}

impl ResourceLayout {
    pub fn new(sets: Vec<vk::DescriptorSetLayout>, constants: Vec<vk::PushConstantRange>) -> ResourceLayout {
        ResourceLayout {
            sets,
            constants,
        }
    }

    pub fn descriptor_sets(&self) -> &[vk::DescriptorSetLayout] {
        &self.sets
    }

    pub fn push_constants(&self) -> &[vk::PushConstantRange] {
        &self.constants
    }
}


pub struct Pipeline<'ctx> {
    context: &'ctx DeviceContext,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    binding_table: ShaderBindingTable<'ctx>,
}

impl<'ctx> Pipeline<'ctx> {

    pub unsafe fn new(
        context: &'ctx DeviceContext,
        resource_layout: &ResourceLayout,
        shader_groups: &[ShaderGroup<'ctx, '_>],
        binding_table_desc: ShaderBindingTableDescription,
    ) -> VkResult<Self> {
        
        let mut stages = vec![];
        let mut stage_infos = vec![];
        let mut group_infos = vec![]; 

        for &shader_group in shader_groups {
            match shader_group {
                ShaderGroup::Raygen { raygen } => {
                    stages.push(raygen);
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .module(raygen.module())
                            .name(&raygen.entry_point)
                            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                            .build()
                    );
                    let stage_index = stage_infos.len() - 1;

                    group_infos.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(stage_index as u32)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .build()
                    );
                },

                ShaderGroup::Miss { miss } => {
                    stages.push(miss);
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .module(miss.module())
                            .name(&miss.entry_point)
                            .stage(vk::ShaderStageFlags::MISS_KHR)
                            .build()
                    );
                    let stage_index = stage_infos.len() - 1;

                    group_infos.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(stage_index as u32)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .build()
                    );
                },

                ShaderGroup::TriangleHit { closest_hit } => {
                    stages.push(closest_hit);
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .module(closest_hit.module())
                            .name(&closest_hit.entry_point)
                            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                            .build()
                    );
                    let closest_hit_stage_index = stage_infos.len() - 1;

                    group_infos.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .general_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(closest_hit_stage_index as u32)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .build()
                    );
                },

                ShaderGroup::Callable { callable } => {
                    stages.push(callable);
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .module(callable.module())
                            .name(&callable.entry_point)
                            .stage(vk::ShaderStageFlags::CALLABLE_KHR)
                            .build()
                    );
                    let stage_index = stage_infos.len() - 1;

                    group_infos.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(stage_index as u32)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .build()
                    );
                }
            }
        }

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&resource_layout.sets)
            .push_constant_ranges(&resource_layout.constants);

        let layout = context.create_pipeline_layout(&layout_info)?;
        
        let info = vk::RayTracingPipelineCreateInfoKHR::builder()
            .stages(&stage_infos)
            .groups(&group_infos)
            .max_pipeline_ray_recursion_depth(3)
            .layout(layout)
            .build();

        let pipeline = context.create_ray_tracing_pipelines(None, None, &[info])?[0];

        let handle_data = context.get_ray_tracing_shader_group_handles(pipeline, 0, group_infos.len() as u32)?;

        let binding_table = binding_table_desc.build(context, &handle_data)?;

        Ok(Pipeline {
            context,
            layout,
            pipeline,
            binding_table,
        })
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn binding_table(&self) -> &ShaderBindingTable<'ctx> {
        &self.binding_table
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl<'a> Drop for Pipeline<'a> {
    fn drop(&mut self) {
        unsafe {
            self.context.destroy_pipeline(self.pipeline);
            self.context.destroy_pipeline_layout(self.layout);
        }
    }
}
