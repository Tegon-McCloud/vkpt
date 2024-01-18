use std::borrow::Cow;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use ash::prelude::VkResult;
use ash::vk;
use ash::extensions::{ext, khr};

pub struct DeviceExtensions {
    pub acceleration_structure: khr::AccelerationStructure,
    pub ray_tracing_pipeline: khr::RayTracingPipeline,
}

pub struct PhysicalDevice {
    pub inner: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    pub ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
}

pub struct DeviceContext {
    #[allow(unused)]
    entry: ash::Entry,
    #[allow(unused)]
    instance: ash::Instance,
    #[allow(unused)]
    debug_utils: ext::DebugUtils,

    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: PhysicalDevice,
    queue_family: u32,
    device: ash::Device,
    extensions: DeviceExtensions,
    queue: vk::Queue,

    allocator: MaybeUninit<Mutex<gpu_allocator::vulkan::Allocator>>,

    command_pool: vk::CommandPool,
}

impl DeviceContext {

    pub fn new() -> VkResult<Self> {
        
        // let entry = unsafe { ash::Entry::load().expect("Failed to load vulkan") };
        let entry = ash::Entry::linked();

        let (instance, debug_utils, debug_messenger) = Self::create_instance(&entry)?;
        let (physical_device, queue_family) = Self::pick_physical_device(&instance)?.expect("no suitable physical device found");
        let (device, extensions, queue) = Self::create_device(&instance, physical_device, queue_family)?;
        let command_pool = Self::create_command_pool(&device, queue_family)?;
        let physical_device = Self::wrap_physical_device(&instance, physical_device);

        
        let allocator = gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device.inner,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
            buffer_device_address: true,
        }).expect("failed to create device memory allocator");

        Ok(Self {   
            entry,
            instance,
            debug_utils,
            
            debug_messenger,
            physical_device,
            queue_family,
            device,
            extensions,
            queue,

            allocator: MaybeUninit::new(Mutex::new(allocator)),

            command_pool,
        })
    }

    fn create_instance(entry: &ash::Entry) -> VkResult<(ash::Instance, ext::DebugUtils, vk::DebugUtilsMessengerEXT)> {
        
        let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::INFO)
            .pfn_user_callback(Some(Self::debug_callback))
            .build();

        let app_name = CStr::from_bytes_until_nul(b"VKPT\0").unwrap();
        
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0))
            .build();


        let extension_names = [
            ext::DebugUtils::name().as_ptr()
        ];
        
        let layer_names = [
            CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap().as_ptr()
        ];
        
        let info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names)
            .flags(vk::InstanceCreateFlags::default())
            .push_next(&mut debug_messenger_info);

        let instance = unsafe { entry.create_instance(&info, None)? };

        let debug_utils = ext::DebugUtils::new(&entry, &instance);
        let debug_messenger = unsafe { debug_utils.create_debug_utils_messenger(&debug_messenger_info, None)? };

        Ok((instance, debug_utils, debug_messenger))
    }


    unsafe extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        ty: vk::DebugUtilsMessageTypeFlagsEXT,
        callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {

        let message = if !(*callback_data).p_message.is_null() {
            CStr::from_ptr((*callback_data).p_message).to_string_lossy()
        } else {
            Cow::Borrowed("")
        };

        let message_id_name = if !(*callback_data).p_message_id_name.is_null() {
            CStr::from_ptr((*callback_data).p_message_id_name).to_string_lossy()
        } else {
            Cow::Borrowed("")
        };

        let message_id_number = (*callback_data).message_id_number;

        let text = format!("[{:?}:{:?}] {} ({}): {}", ty, severity, message_id_name, message_id_number, message);

        println!("{}\n", text);
        
        vk::FALSE
    }
    
    fn pick_queue_family(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {

        let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        
        queue_family_properties.iter()
            .enumerate()
            .filter(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER))
            .map(|(i, _)| i as u32)
            .next()
    }

    fn wrap_physical_device(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> PhysicalDevice {

        let mut acceleration_structure_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut ray_tracing_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

        let mut properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut acceleration_structure_properties)
            .push_next(&mut ray_tracing_pipeline_properties);
        
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut properties2);
        }

        PhysicalDevice {
            inner: physical_device,
            properties: properties2.properties,
            acceleration_structure_properties,
            ray_tracing_pipeline_properties,
        }
    }

    fn rate_physical_device(_instance: &ash::Instance, _physical_device: vk::PhysicalDevice, _queue_family: u32) -> u32 {
        1
    }

    fn pick_physical_device(instance: &ash::Instance) -> VkResult<Option<(vk::PhysicalDevice, u32)>> {

        let mut best: Option<(vk::PhysicalDevice, u32)> = None;
        let mut best_score = 0;

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        for physical_device in physical_devices {

            let queue_family = Self::pick_queue_family(instance, physical_device);
            if queue_family.is_none() {
                continue;
            }
            let queue_family = queue_family.unwrap();

            let score = Self::rate_physical_device(instance, physical_device, queue_family);

            if score > best_score {
                best = Some((physical_device, queue_family));
                best_score = score;
            }
        }

        Ok(best)
    }

    fn create_device(instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family: u32) -> VkResult<(ash::Device, DeviceExtensions, vk::Queue)> {
        let priority = [1.0];

        let queue_create_info =  vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&priority)
            .build();

        let extension_names = [
            khr::DeferredHostOperations::name().as_ptr(),
            khr::AccelerationStructure::name().as_ptr(),
            khr::RayTracingPipeline::name().as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures::builder()
            .shader_int64(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            // .descriptor_binding_partially_bound(true)
            // .descriptor_indexing(true)
            .build();

        let mut features13 = vk::PhysicalDeviceVulkan13Features::builder()
            .synchronization2(true)
            .build();

        let mut accel_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
            .acceleration_structure(true)
            .build();

        let mut ray_tracing_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder()
            .ray_tracing_pipeline(true)
            .build();

        // let extension_properties = unsafe { instance.enumerate_device_extension_properties(physical_device)? };

        // for props in extension_properties {
            
        //     let name_buffer = props.extension_name.iter().map(|c| *c as u8).chain(std::iter::once(b'\0')).collect::<Vec<_>>();
        //     let name = CStr::from_bytes_until_nul(&name_buffer);

        //     println!("{:?}", name);
        // }

        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&extension_names)
            .enabled_features(&features)
            .push_next(&mut features12)
            .push_next(&mut features13)
            .push_next(&mut accel_features)
            .push_next(&mut ray_tracing_features);
        
        let device = unsafe { instance.create_device(physical_device, &info, None)? };

        let extensions = DeviceExtensions {
            acceleration_structure: khr::AccelerationStructure::new(instance, &device),
            ray_tracing_pipeline: khr::RayTracingPipeline::new(instance, &device),
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok((device, extensions, queue))
    }

    fn create_command_pool(device: &ash::Device, queue_family: u32) -> VkResult<vk::CommandPool> {

        let create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family);

        unsafe { device.create_command_pool(&create_info, None) }
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }
    
    pub fn extensions(&self) -> &DeviceExtensions {
        &self.extensions
    }

    pub fn allocator(&self) -> &Mutex<gpu_allocator::vulkan::Allocator> {
        unsafe { self.allocator.assume_init_ref() }
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn transition_image(&self, cmd_buffer: vk::CommandBuffer, image: vk::Image, current_layout: vk::ImageLayout, new_layout: vk::ImageLayout) {

        let barrier = vk::ImageMemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .old_layout(current_layout)
            .new_layout(new_layout)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .build())
            .image(image)
            .build();

        let dependency = vk::DependencyInfo::builder()
            .image_memory_barriers(std::slice::from_ref(&barrier));
        
        unsafe { self.device.cmd_pipeline_barrier2(cmd_buffer, &dependency) };
    }

    pub unsafe fn create_timeline_semaphore(&self) -> VkResult<vk::Semaphore> {
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let semaphore_info = vk::SemaphoreCreateInfo::builder()
            .push_next(&mut timeline_info);
        
        self.device.create_semaphore(&semaphore_info, None)
    }

    pub unsafe fn destroy_timeline_semaphore(&self, semaphore: vk::Semaphore) {
        self.device.destroy_semaphore(semaphore, None);
    }


    // should only be used for testing since just stalls host until device is finished
    pub unsafe fn execute_commands<F>(&self, record_fn: F) -> VkResult<()> where
        F: FnOnce(vk::CommandBuffer)
    {
        let cmd_buffer_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer = self.device.allocate_command_buffers(&cmd_buffer_info)?[0];

        let semaphore = self.create_timeline_semaphore()?;

        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(cmd_buffer, &cmd_buffer_begin_info)?;

        record_fn(cmd_buffer);

        self.device.end_command_buffer(cmd_buffer)?;

        let command_buffer_submit_info = vk::CommandBufferSubmitInfo::builder()
            .command_buffer(cmd_buffer)
            .build();

        let semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
            .semaphore(semaphore)
            .value(1)
            .build();

        let submit_info = vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&command_buffer_submit_info))
            .signal_semaphore_infos(&[semaphore_submit_info])
            .build();

        self.device.queue_submit2(self.queue, &[submit_info], vk::Fence::null())?;

        let semaphore_wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(std::slice::from_ref(&semaphore))
            .values(&[1]);

        self.device.wait_semaphores(&semaphore_wait_info, 1 << 32)?;
        
        self.destroy_timeline_semaphore(semaphore);
        self.device.free_command_buffers(self.command_pool, std::slice::from_ref(&cmd_buffer));
        
        Ok(())
    }


}


impl DeviceContext {

    // Core

    // Resources
    pub unsafe fn create_buffer(&self, create_info: &vk::BufferCreateInfo) -> VkResult<vk::Buffer> {
        self.device.create_buffer(create_info, None)
    }

    pub unsafe fn destroy_buffer(&self, buffer: vk::Buffer) {
        self.device.destroy_buffer(buffer, None)
    }

    pub unsafe fn get_buffer_memory_requirements(&self, buffer: vk::Buffer) -> vk::MemoryRequirements {
        self.device.get_buffer_memory_requirements(buffer)
    }

    pub unsafe fn bind_buffer_memory2(&self, bind_infos: &[vk::BindBufferMemoryInfo]) -> VkResult<()> {
        self.device.bind_buffer_memory2(bind_infos)
    }

    pub unsafe fn flush_mapped_memory_ranges(&self, ranges: &[vk::MappedMemoryRange]) -> VkResult<()> {
        self.device.flush_mapped_memory_ranges(ranges)
    }

    pub unsafe fn invalidate_mapped_memory_ranges(&self, ranges: &[vk::MappedMemoryRange]) -> VkResult<()> {
        self.device.invalidate_mapped_memory_ranges(ranges)
    }

    pub unsafe fn get_buffer_device_address(&self, info: &vk::BufferDeviceAddressInfo) -> vk::DeviceAddress {
        self.device.get_buffer_device_address(info)
    }

    // 
    


}


impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.allocator.assume_init_drop();
            self.device.destroy_device(None);

            self.debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
