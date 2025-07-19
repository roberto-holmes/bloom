__import structs;
__import constants;
__import rng;
__import helpers;

struct Instance {
	AccelerationStructureInstanceKHR asi;
	float4x4 base_transform;  // Investigate if we can use float4x3 instead (to save on space)
	float4x4 world_transform;
	uint32_t id;
	uint32_t pad0;
	uint32_t pad1;
	uint32_t pad2;
}

struct AccelerationStructureInstanceKHR {
	float4x3 transform;
	uint32_t index_mask;
	uint32_t offset_flags;
	uint64_t reference;
}

struct PushConstants {
	uint64_t time_stamp_ns;
	uint64_t* instances;
	uint32_t instance_count;
}

// [[vk::binding(0, 0)]]
// RWTexture2D<float4> ocean_map;	// TODO: Consider replacing with a float2 texture
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(8, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	if (id.x >= pc.instance_count) {
		return;
	}

	Instance* inst = ((Instance*)*(pc.instances + id.x));

	// if (pc.time_stamp_ns < 25_000_000) {
	//  printf("%u: %u [%lu] %f\n", id.x, inst.id, (uint64_t)inst, inst.asi.transform[3][1]);
	// }

	if (inst.id == 5) {
		inst.world_transform[1][3] = float(pc.time_stamp_ns / float(1000_000_000));
	}
	// Acceleration structure uses opposite matrix layout
	inst.asi.transform = float4x3(transpose(mul(inst.world_transform, inst.base_transform)));
}
