use std::{ffi::{CStr, CString}, fs::File, io::{Read, Seek, SeekFrom}, path::Path};

use ash::{prelude::VkResult, vk};
use itertools::Itertools;

use crate::{context::DeviceContext, util};

// #[derive(Debug, Hash)]
// pub enum ShaderModuleSource {
//     File(PathBuf),
//     Code(Vec<u32>),
// }

// pub enum ShaderModuleSourceRef<'a, P> {
//     File(&'a P),
//     Code(&'a [u32]),
// }


pub struct Shader<'a> {
    context: &'a DeviceContext,
    module: vk::ShaderModule,
    entry_point: CString,
    resource_layout: Vec<vk::DescriptorSetLayout>,
}

impl<'a> Shader<'a> {
    pub unsafe fn new<P: AsRef<Path>>(
        context: &'a DeviceContext,
        spv_path: P,
        entry_point: CString,
        resource_layout: Vec<vk::DescriptorSetLayout>,
    ) -> VkResult<Self> {
        let spv_code = Self::read_spv(spv_path.as_ref());
        let info = vk::ShaderModuleCreateInfo::builder()
            .code(&spv_code);
        let module = context.create_shader_module(&info)?;

        Ok(Self {
            context,
            module,
            entry_point,
            resource_layout,
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

    pub fn module(&self) -> vk::ShaderModule {
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

// pub struct ShaderModuleCache<'a> {
//     context: &'a DeviceContext,

//     map: HashMap<ShaderModuleSource, vk::ShaderModule>
// }

// impl<'a> ShaderModuleCache<'a> {

//     pub fn new(context: &'a DeviceContext) -> Self {
//         Self {
//             context,
//             map: HashMap::new(),
//         }
//     }

//     pub unsafe fn resolve<P: AsRef<Path>>(&mut self, spv_path: P) -> VkResult<vk::ShaderModule> {
//         let spv_path = spv_path.as_ref();

//         let canonical_spv_path = spv_path
//             .canonicalize()
//             .ok()
//             .unwrap_or_else(|| spv_path.to_path_buf());

//         let shader_module = match self.map.entry(canonical_spv_path) {
            
//             Entry::Occupied(entry) => {
//                 *entry.get()
//             },

//             Entry::Vacant(entry) => {
//                 let mut file = File::open(entry.key()).expect("failed to open SPIR-V file");
                
//                 let spv_size = file.metadata().unwrap().file_size() as usize;
//                 let u32_buffer_size = spv_size.div_ceil(std::mem::size_of::<u32>());
            
//                 let mut buffer = vec![0u32; u32_buffer_size];
                
//                 file.read(util::slice_as_mut_u8_slice(buffer.as_mut_slice())).expect("failed to read SPIR-V file");
            
//                 let info = vk::ShaderModuleCreateInfo::builder()
//                     .code(&buffer);
            
//                 let shader_module = self.context.create_shader_module(&info)?;
                
//                 *entry.insert(shader_module)
//             },
//         };

//         Ok(shader_module)
//     }
// }

// pub struct PipelineStage {
//     stage: vk::ShaderStageFlags,
//     shader_builder: ShaderBuilder,
// }

// pub struct PipelineBuilder {
//     layout: vk::PipelineLayout,
//     stages: Vec<vk::PipelineShaderStageCreateInfo>,
//     groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
// }

// impl PipelineBuilder {
//     pub fn new(layout: vk::PipelineLayout) -> Self {
//         Self {
//             layout,
//             stages: Vec::new(),
//             groups: Vec::new(),
//         }
//     }

//     fn add_stage(&mut self, shader: ShaderReference<'_>, stage: vk::ShaderStageFlags) -> u32 {
//         self.stages.push(vk::PipelineShaderStageCreateInfo::builder()
//             .stage(stage)
//             .module(shader.module)
//             .name(shader.entry)
//             .build());

//         (self.stages.len() - 1) as u32
//     }

//     // pub fn add_raygen_stage(&mut self, module: vk::ShaderModule, entry: &CStr) -> u32 {
//     //     self.add_stage(module, entry, vk::ShaderStageFlags::RAYGEN_KHR)
//     // }

//     // pub fn add_miss_stage(&mut self, module: vk::ShaderModule, entry: &CStr) -> u32 {
//     //     self.add_stage(module, entry, vk::ShaderStageFlags::MISS_KHR)
//     // }

//     // pub fn add_closest_hit_stage(&mut self, module: vk::ShaderModule, entry: &CStr) -> u32 {
//     //     self.add_stage(module, entry, vk::ShaderStageFlags::CLOSEST_HIT_KHR)
//     // }

//     // pub fn add_callable_stage(&mut self, module: vk::ShaderModule, entry: &CStr) -> u32 {
//     //     self.add_stage(module, entry, vk::ShaderStageFlags::CALLABLE_KHR)
//     // }

//     fn add_general_group(&mut self, shader: ShaderReference<'_>, stage: vk::ShaderStageFlags) -> u32 {
//         let stage = self.add_stage(shader, stage);

//         self.groups.push(vk::RayTracingShaderGroupCreateInfoKHR::builder()
//             .any_hit_shader(vk::SHADER_UNUSED_KHR)
//             .closest_hit_shader(vk::SHADER_UNUSED_KHR)
//             .intersection_shader(vk::SHADER_UNUSED_KHR)
//             .general_shader(stage)
//             .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
//             .build());

//         (self.groups.len() - 1) as u32
//     }

//     pub fn add_raygen_group(&mut self, raygen: ShaderReference<'_>) -> u32 {
//         self.add_general_group(raygen, vk::ShaderStageFlags::RAYGEN_KHR)
//     }

//     pub fn add_miss_group(&mut self, miss: ShaderReference<'_>) -> u32 {
//         self.add_general_group(miss, vk::ShaderStageFlags::MISS_KHR)
//     }

//     pub fn add_callable_group(&mut self, callable: ShaderReference<'_>) -> u32 {
//         self.add_general_group(callable, vk::ShaderStageFlags::CALLABLE_KHR)
//     }

//     pub fn add_triangles_hit_group(&mut self, closest_hit: ShaderReference) -> u32 {
//         let closest_hit_stage = self.add_stage(closest_hit, vk::ShaderStageFlags::CLOSEST_HIT_KHR);

//         self.groups.push(vk::RayTracingShaderGroupCreateInfoKHR::builder()
//             .any_hit_shader(vk::SHADER_UNUSED_KHR)
//             .closest_hit_shader(closest_hit_stage)
//             .intersection_shader(vk::SHADER_UNUSED_KHR)
//             .general_shader(vk::SHADER_UNUSED_KHR)
//             .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
//             .build());

//         (self.groups.len() - 1) as u32
//     }

//     pub unsafe fn build<'a>(&self, context: &'a DeviceContext) -> VkResult<Pipeline<'a>> {
        
//         let pipeline = self.build_pipeline()?;

//         let handle_data = context.get_ray_tracing_shader_group_handles(pipeline, 0, self.groups.len() as u32)?;

//         Ok(Pipeline {
//             context,
//             pipeline,

//             handle_data,
//         })
//     }

//     unsafe fn build_pipeline(&self, context: &DeviceContext) -> VkResult<vk::Pipeline> {

//         let info = vk::RayTracingPipelineCreateInfoKHR::builder();

//         context.create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), infos)
        
//     } 

// }
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

pub struct ShaderGroupHandle(usize);

pub struct PipelineDescription<'ctx, 'a> {
    shader_groups: Vec<ShaderGroup<'ctx, 'a>>,
}

impl<'ctx, 'a> PipelineDescription<'ctx, 'a> {

    pub fn new() -> Self {
        Self {
            shader_groups: Vec::new(),
        }
    }
    
    pub fn add_group(&mut self, group: ShaderGroup<'ctx, 'a>) -> ShaderGroupHandle {
        self.shader_groups.push(group);
        ShaderGroupHandle(self.shader_groups.len() - 1)
    }

    pub unsafe fn build(&self, context: &'ctx DeviceContext) -> VkResult<Pipeline<'ctx>> {

        let mut stages = vec![];
        let mut stage_infos = vec![];
        let mut group_infos = vec![]; 

        for &shader_group in &self.shader_groups {
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
        
        Pipeline::new(context, &stages, &stage_infos, &group_infos, None)
    }

}

pub struct Pipeline<'ctx> {
    context: &'ctx DeviceContext,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    handle_data: Vec<u8>,
}

impl<'a> Pipeline<'a> {

    pub unsafe fn new(
        context: &'a DeviceContext,
        stages: &[&Shader<'a>],
        stage_infos: &[vk::PipelineShaderStageCreateInfo],
        group_infos: &[vk::RayTracingShaderGroupCreateInfoKHR],
        libraries: Option<(&[vk::Pipeline], vk::RayTracingPipelineInterfaceCreateInfoKHR)>,
    ) -> VkResult<Self> {

        let set_layouts = stages.iter()
            .map(|stage| stage.resource_layout.iter().copied())
            .flatten()
            .collect_vec();

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts);

        let layout = context.create_pipeline_layout(&layout_info)?;

        let (libraries, interface_info) = libraries.unwrap_or_default();

        let library_info = vk::PipelineLibraryCreateInfoKHR::builder()
            .libraries(&libraries)
            .build();

        let info = vk::RayTracingPipelineCreateInfoKHR::builder()
            .stages(stage_infos)
            .groups(group_infos)
            .max_pipeline_ray_recursion_depth(3)
            .library_info(&library_info)
            .library_interface(&interface_info)
            .layout(layout)
            .build();

        let pipeline = context.create_ray_tracing_pipelines(None, None, &[info])?[0];

        let handle_data = context.get_ray_tracing_shader_group_handles(pipeline, 0, group_infos.len() as u32)?;

        Ok(Self {
            context,
            layout,
            pipeline,
            handle_data,
        })
    }

    pub fn group_count(&self) -> u32 {
        (self.handle_data.len() / self.handle_stride()) as u32
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn get_group_handle(&self, index: u32) -> &[u8] {
        let stride = self.handle_stride();
        &self.handle_data[stride * index as usize..stride * (index + 1) as usize]
    }

    fn handle_stride(&self) -> usize {
        self.context.physical_device().ray_tracing_pipeline_properties.shader_group_handle_size as usize
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
