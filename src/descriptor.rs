use ash::vk;

pub struct StandaloneDescriptorSet {
    inner: vk::DescriptorSet,
    pool: vk::DescriptorPool,
}


